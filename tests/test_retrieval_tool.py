"""Tests for the retrieval tool (Feature 6) — lets the model ask back for
content Feature 2/3 collapsed, instead of guessing. `call_upstream` is
mocked here since these are pure loop-mechanics tests (does the bounded
retry loop resolve correctly, handle a missing key gracefully, and cap out
safely) — a real end-to-end check against OpenRouter covers whether the
model actually uses the tool sensibly."""
from __future__ import annotations

import json

import wrapper_server
from wrapper_server import _MAX_RETRIEVAL_ROUNDS, _resolve_with_retrieval


def _tool_call_response(key: str, call_id: str = "call_1") -> dict:
    return {
        "choices": [{
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": "shapeshifter_expand", "arguments": json.dumps({"key": key})},
                }],
            },
        }],
    }


def _final_response(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def test_resolve_with_retrieval_answers_tool_call_and_returns_final_answer(monkeypatch):
    calls = []

    async def fake_call_upstream(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _tool_call_response("app.py"), 100.0
        return _final_response("The file has a divide() bug fixed."), 150.0

    monkeypatch.setattr(wrapper_server, "call_upstream", fake_call_upstream)

    import asyncio
    resp, total_latency, rounds = asyncio.run(_resolve_with_retrieval(
        base_url="https://fake", api_key="k", model="m",
        messages=[{"role": "user", "content": "fix the bug"}],
        temperature=0.2, max_tokens=100, extra={},
        retrieval_map={"app.py": "def divide(a, b):\n    return a / b\n"},
    ))

    assert resp["choices"][0]["message"]["content"] == "The file has a divide() bug fixed."
    assert total_latency == 250.0
    assert rounds == 1  # resolved on the second upstream call (round index 1)
    assert len(calls) == 2
    # the tool result fed back to the model must contain the real content
    tool_msg = next(m for m in calls[1]["messages"] if m.get("role") == "tool")
    assert "def divide" in tool_msg["content"]


def test_resolve_with_retrieval_handles_missing_key_gracefully(monkeypatch):
    calls = []

    async def fake_call_upstream(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _tool_call_response("nonexistent.py"), 100.0
        return _final_response("done"), 100.0

    monkeypatch.setattr(wrapper_server, "call_upstream", fake_call_upstream)

    import asyncio
    resp, _, _ = asyncio.run(_resolve_with_retrieval(
        base_url="https://fake", api_key="k", model="m",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2, max_tokens=100, extra={},
        retrieval_map={"app.py": "content"},
    ))

    tool_msg = next(m for m in calls[1]["messages"] if m.get("role") == "tool")
    assert "No collapsed content found" in tool_msg["content"]
    assert resp["choices"][0]["message"]["content"] == "done"  # loop still completes normally


def test_resolve_with_retrieval_caps_out_and_forces_final_answer(monkeypatch):
    calls = []

    async def fake_call_upstream(**kwargs):
        calls.append(kwargs)
        # Always asks for more, even after the cap — the loop must stop
        # calling it with tools available and force a plain answer instead.
        if "tools" not in kwargs.get("extra_params", {}):
            return _final_response("forced final answer"), 50.0
        return _tool_call_response("app.py", call_id=f"call_{len(calls)}"), 50.0

    monkeypatch.setattr(wrapper_server, "call_upstream", fake_call_upstream)

    import asyncio
    resp, total_latency, rounds = asyncio.run(_resolve_with_retrieval(
        base_url="https://fake", api_key="k", model="m",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2, max_tokens=100, extra={},
        retrieval_map={"app.py": "content"},
    ))

    assert rounds == _MAX_RETRIEVAL_ROUNDS
    assert resp["choices"][0]["message"]["content"] == "forced final answer"
    # _MAX_RETRIEVAL_ROUNDS calls with the tool available, plus 1 final call without it
    assert len(calls) == _MAX_RETRIEVAL_ROUNDS + 1
    assert "tools" not in calls[-1]["extra_params"]


def test_resolve_with_retrieval_skips_loop_entirely_when_no_tool_call(monkeypatch):
    async def fake_call_upstream(**kwargs):
        return _final_response("plain answer, no tool needed"), 120.0

    monkeypatch.setattr(wrapper_server, "call_upstream", fake_call_upstream)

    import asyncio
    resp, total_latency, rounds = asyncio.run(_resolve_with_retrieval(
        base_url="https://fake", api_key="k", model="m",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2, max_tokens=100, extra={},
        retrieval_map={"app.py": "content"},
    ))

    assert rounds == 0
    assert total_latency == 120.0
    assert resp["choices"][0]["message"]["content"] == "plain answer, no tool needed"
