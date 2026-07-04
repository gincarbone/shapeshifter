"""Tests for the request-shaping logic in wrapper_server.py — agentic/tool-call
detection, system-prompt preservation, trailing-message preservation, and
extra-param passthrough. These are the pipeline bugs that used to silently
corrupt tool-calling sessions (see conversation history / CHANGELOG)."""
from __future__ import annotations

from wrapper_server import (
    _build_compressed_messages,
    _build_tool_call_keys,
    _dedupe_repeated_tool_calls,
    _extra_params,
    _is_agentic,
)


# ---------------------------------------------------------------------------
# Agentic / tool-call detection
# ---------------------------------------------------------------------------

def test_plain_chat_is_not_agentic():
    messages = [{"role": "user", "content": "hello"}]
    assert _is_agentic(messages, {}) is False


def test_tools_in_body_marks_agentic():
    messages = [{"role": "user", "content": "what's the weather"}]
    body = {"tools": [{"type": "function", "function": {"name": "get_weather"}}]}
    assert _is_agentic(messages, body) is True


def test_tool_choice_in_body_marks_agentic():
    messages = [{"role": "user", "content": "hi"}]
    assert _is_agentic(messages, {"tool_choice": "auto"}) is True


def test_assistant_tool_calls_marks_agentic():
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function",
                                                                "function": {"name": "get_weather", "arguments": "{}"}}]},
    ]
    assert _is_agentic(messages, {}) is True


def test_tool_role_message_marks_agentic():
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "sunny"},
    ]
    assert _is_agentic(messages, {}) is True


def test_anthropic_style_content_blocks_marks_agentic():
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "x", "input": {}}]},
    ]
    assert _is_agentic(messages, {}) is True


# ---------------------------------------------------------------------------
# System-prompt preservation + trailing-message preservation
# (previously: client system message was silently discarded, and only the
# single last user message was kept — anything after it was dropped)
# ---------------------------------------------------------------------------

def test_client_system_message_is_preserved_and_combined():
    messages = [
        {"role": "system", "content": "CLIENT_MARKER: always say hi"},
        {"role": "user", "content": "generate a function"},
    ]
    new_messages, _, _, _ = _build_compressed_messages(messages, "hybrid")
    system_msg = next(m for m in new_messages if m["role"] == "system")
    assert "CLIENT_MARKER: always say hi" in system_msg["content"]


def test_no_client_system_message_still_works():
    messages = [{"role": "user", "content": "generate a function"}]
    new_messages, _, _, _ = _build_compressed_messages(messages, "hybrid")
    assert new_messages[0]["role"] == "system"
    assert new_messages[-1]["content"] == "generate a function"


def test_messages_after_last_user_turn_are_not_dropped():
    # Regression test: previously only messages[last_user_idx] was kept as
    # "current", silently dropping anything appended after it.
    messages = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "```python\nprint(1)\n```"},
        {"role": "user", "content": "second turn"},
        {"role": "assistant", "content": "trailing note after the last user message"},
    ]
    new_messages, _, _, _ = _build_compressed_messages(messages, "hybrid")
    tail_contents = [m.get("content") for m in new_messages]
    assert "second turn" in tail_contents
    assert "trailing note after the last user message" in tail_contents


# ---------------------------------------------------------------------------
# Extra params passthrough (tools/tool_choice/etc. must reach the upstream)
# ---------------------------------------------------------------------------

def test_extra_params_forwards_tools_and_unknown_fields():
    body = {
        "model": "gpt-4o", "messages": [], "temperature": 0.2, "max_tokens": 100,
        "stream": True, "context_mode": "hybrid",
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": "auto", "top_p": 0.9, "seed": 42,
    }
    extra = _extra_params(body)
    assert extra == {
        "tools": body["tools"], "tool_choice": "auto", "top_p": 0.9, "seed": 42,
    }


def test_extra_params_empty_when_only_handled_fields_present():
    body = {"model": "gpt-4o", "messages": [], "temperature": 0.2, "max_tokens": 100}
    assert _extra_params(body) == {}


# ---------------------------------------------------------------------------
# Tool-call deduplication (Feature 1 + Feature 7 — generalized beyond file
# reads to any repeated tool call with identical arguments). Only removes an
# exact, unmodified-or-superseded LATER-resolvable occurrence — never
# anything but `tool` messages, and the last occurrence always stays full.
# ---------------------------------------------------------------------------

def _read_file_call(call_id, path):
    return {"id": call_id, "type": "function",
            "function": {"name": "read_file", "arguments": f'{{"path": "{path}"}}'}}


def _generic_call(call_id, name, args_json):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args_json}}


def test_build_tool_call_keys_extracts_path_from_read_file_call():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [_read_file_call("1", "src/app.py")]},
    ]
    assert _build_tool_call_keys(messages) == {"1": ("file:src/app.py", "src/app.py")}


def test_build_tool_call_keys_tracks_non_read_tools_too():
    # Feature 7: generalized beyond file reads — any tool call gets a key.
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "execute_command", "arguments": '{"command": "ls"}'}},
        ]},
    ]
    key, label = _build_tool_call_keys(messages)["1"]
    assert label == "execute_command"
    assert key.startswith("call:execute_command:")


def test_build_tool_call_keys_recognizes_alternate_path_arg_names():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "get_file_content", "arguments": '{"filename": "x.py"}'}},
        ]},
    ]
    assert _build_tool_call_keys(messages) == {"1": ("file:x.py", "x.py")}


def test_build_tool_call_keys_different_arguments_get_different_keys():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "execute_command", "arguments": '{"command": "ls"}'}},
            {"id": "2", "type": "function", "function": {"name": "execute_command", "arguments": '{"command": "pwd"}'}},
        ]},
    ]
    keys = _build_tool_call_keys(messages)
    assert keys["1"][0] != keys["2"][0]  # different arguments -> different dedup key


def test_dedupe_replaces_earlier_identical_read_keeps_last_in_full():
    # Realistic file size — the size guard only dedupes when the marker is
    # actually smaller than what it replaces.
    file_content = "def add(a, b):\n    return a + b\n" * 20
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [_read_file_call("1", "app.py")]},
        {"role": "tool", "tool_call_id": "1", "content": file_content},
        {"role": "user", "content": "now add sub()"},
        {"role": "assistant", "content": None, "tool_calls": [_read_file_call("2", "app.py")]},
        {"role": "tool", "tool_call_id": "2", "content": file_content},  # unmodified re-read
    ]
    result = _dedupe_repeated_tool_calls(messages)
    first_tool_msg = next(m for m in result if m.get("tool_call_id") == "1")
    last_tool_msg = next(m for m in result if m.get("tool_call_id") == "2")
    assert first_tool_msg["content"] != file_content
    assert "unchanged" in first_tool_msg["content"]
    assert last_tool_msg["content"] == file_content  # last occurrence always kept in full


def test_dedupe_three_reads_only_last_stays_full():
    v1 = "def add(a, b):\n    return a + b\n" * 20
    v2 = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n" * 20
    v3 = v2 + "\n# final tweak\n" * 20
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [_read_file_call("1", "app.py")]},
        {"role": "tool", "tool_call_id": "1", "content": v1},
        {"role": "assistant", "content": None, "tool_calls": [_read_file_call("2", "app.py")]},
        {"role": "tool", "tool_call_id": "2", "content": v2},
        {"role": "assistant", "content": None, "tool_calls": [_read_file_call("3", "app.py")]},
        {"role": "tool", "tool_call_id": "3", "content": v3},
    ]
    result = _dedupe_repeated_tool_calls(messages)
    by_id = {m["tool_call_id"]: m["content"] for m in result if m.get("role") == "tool"}
    assert by_id["1"] != v1 and "superseded" in by_id["1"]
    assert by_id["2"] != v2 and "superseded" in by_id["2"]
    assert by_id["3"] == v3  # only the last read stays full


def test_dedupe_size_guard_skips_files_smaller_than_the_marker():
    # For a tiny file, the marker text would be LARGER than what it replaces —
    # never do that, it would make things worse, not better.
    tiny_content = "x = 1\n"
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [_read_file_call("1", "x.py")]},
        {"role": "tool", "tool_call_id": "1", "content": tiny_content},
        {"role": "assistant", "content": None, "tool_calls": [_read_file_call("2", "x.py")]},
        {"role": "tool", "tool_call_id": "2", "content": tiny_content},
    ]
    result = _dedupe_repeated_tool_calls(messages)
    assert result == messages


def test_dedupe_replaces_earlier_differing_read_too_keeps_last_in_full():
    # Latest-wins: an earlier read that's since been superseded (file
    # actually changed) is also collapsed, not just an identical re-read —
    # only the LAST read of a path needs to stay in full to act on it now.
    old_content = ("def add(a, b):\n    return a + b\n" * 20)
    new_content = ("def add(a, b):\n    return a + b + 1\n" * 20)
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [_read_file_call("1", "app.py")]},
        {"role": "tool", "tool_call_id": "1", "content": old_content},
        {"role": "assistant", "content": None, "tool_calls": [_read_file_call("2", "app.py")]},
        {"role": "tool", "tool_call_id": "2", "content": new_content},  # file changed since
    ]
    result = _dedupe_repeated_tool_calls(messages)
    first_tool_msg = next(m for m in result if m.get("tool_call_id") == "1")
    last_tool_msg = next(m for m in result if m.get("tool_call_id") == "2")
    assert first_tool_msg["content"] != old_content
    assert "superseded" in first_tool_msg["content"]
    assert last_tool_msg["content"] == new_content  # last occurrence always kept in full


def test_dedupe_size_guard_applies_to_differing_reads_too():
    # Same size guard as the identical case: a tiny file's marker would be
    # bigger than the content it replaces, so leave it alone even though the
    # content differs between reads.
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [_read_file_call("1", "x.py")]},
        {"role": "tool", "tool_call_id": "1", "content": "x = 1\n"},
        {"role": "assistant", "content": None, "tool_calls": [_read_file_call("2", "x.py")]},
        {"role": "tool", "tool_call_id": "2", "content": "x = 2\n"},
    ]
    result = _dedupe_repeated_tool_calls(messages)
    assert result == messages


def test_dedupe_now_covers_non_file_tool_calls_too():
    # Feature 7: a repeated execute_command with the SAME arguments and
    # identical output is exactly as safe to dedupe as a repeated file read.
    output = "PASS test_add\nPASS test_sub\nPASS test_mul\n" * 10  # realistic size, bypasses the guard
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [_generic_call("1", "execute_command", '{"command": "pytest"}')]},
        {"role": "tool", "tool_call_id": "1", "content": output},
        {"role": "assistant", "content": None, "tool_calls": [_generic_call("2", "execute_command", '{"command": "pytest"}')]},
        {"role": "tool", "tool_call_id": "2", "content": output},  # identical re-run, same command
    ]
    result = _dedupe_repeated_tool_calls(messages)
    first_tool_msg = next(m for m in result if m.get("tool_call_id") == "1")
    last_tool_msg = next(m for m in result if m.get("tool_call_id") == "2")
    assert first_tool_msg["content"] != output
    assert "unchanged" in first_tool_msg["content"] and "execute_command" in first_tool_msg["content"]
    assert last_tool_msg["content"] == output  # last occurrence always kept in full


def test_dedupe_non_file_tool_calls_with_different_arguments_stay_untouched():
    # Different command -> different dedup key -> not a dupe at all.
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [_generic_call("1", "execute_command", '{"command": "pytest"}')]},
        {"role": "tool", "tool_call_id": "1", "content": "PASS" * 50},
        {"role": "assistant", "content": None, "tool_calls": [_generic_call("2", "execute_command", '{"command": "ls -la"}')]},
        {"role": "tool", "tool_call_id": "2", "content": "PASS" * 50},
    ]
    result = _dedupe_repeated_tool_calls(messages)
    assert result == messages  # different arguments -> not the same call, never dedupe


def test_dedupe_does_not_touch_non_tool_messages():
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert _dedupe_repeated_tool_calls(messages) == messages
