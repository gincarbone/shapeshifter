"""Tests for the real streaming SSE parser in llm_client.py — this replaced
the old fake-stream implementation that buffered the whole response before
emitting a single synthetic chunk. Verifies real upstream chunks (including
multi-event buffers and error responses) are parsed and relayed correctly."""
from __future__ import annotations

import httpx
import pytest

import llm_client
from llm_client import stream_upstream

_RealAsyncClient = httpx.AsyncClient  # captured before any monkeypatching


def _mock_client_factory(transport: httpx.MockTransport):
    def factory(*args, **kwargs):
        return _RealAsyncClient(transport=transport)
    return factory


async def _collect(agen):
    return [chunk async for chunk in agen]


def test_stream_upstream_yields_parsed_chunks_in_order(monkeypatch):
    body = (
        b'data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"id":"1","choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n'
        b'data: {"id":"1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}\n\n'
        b'data: [DONE]\n\n'
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, content=body, headers={"content-type": "text/event-stream"},
    ))
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _mock_client_factory(transport))

    import asyncio
    chunks = asyncio.run(_collect(stream_upstream(
        "https://fake.example/v1", "key", "model", [{"role": "user", "content": "hi"}],
    )))

    assert len(chunks) == 3
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[2]["choices"][0]["delta"]["content"] == " world"
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"


def test_stream_upstream_handles_chunk_boundaries_mid_event(monkeypatch):
    """Upstream bytes can arrive split mid-event; the parser must buffer
    until a full '\\n\\n'-terminated event is available before parsing JSON."""
    full = b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    split_at = 20

    def handler(request):
        async def body_iter():
            yield full[:split_at]
            yield full[split_at:]
        return httpx.Response(200, content=body_iter())

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _mock_client_factory(transport))

    import asyncio
    chunks = asyncio.run(_collect(stream_upstream(
        "https://fake.example/v1", "key", "model", [{"role": "user", "content": "hi"}],
    )))
    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["delta"]["content"] == "ok"


def test_stream_upstream_raises_on_http_error(monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(
        401, content=b'{"error":{"message":"unauthorized"}}',
    ))
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _mock_client_factory(transport))

    import asyncio

    async def run():
        async for _ in stream_upstream(
            "https://fake.example/v1", "key", "model", [{"role": "user", "content": "hi"}],
        ):
            pass

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run())


def test_build_payload_forwards_extra_params_but_not_reserved_keys():
    payload = llm_client._build_payload(
        "model-x", [{"role": "user", "content": "hi"}], 0.2, 100,
        extra_params={"tools": [{"type": "function"}], "model": "should-not-override", "top_p": 0.9},
        stream=True,
    )
    assert payload["model"] == "model-x"          # reserved key not overridden by extra_params
    assert payload["tools"] == [{"type": "function"}]
    assert payload["top_p"] == 0.9
    assert payload["stream"] is True
