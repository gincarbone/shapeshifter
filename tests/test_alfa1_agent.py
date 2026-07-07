"""Tests for alfa1_agent.py — the plain-text action protocol (file writes via
fenced code blocks with a filename header, and <alfa1:...> action tags) and
the agent loop. No OpenAI tool-calling is used here on purpose: see the
module docstring in alfa1_agent.py for why (it would make wrapper_server's
_is_agentic bypass ShapeShifter's own input/output token-savings pipeline)."""
from __future__ import annotations

import asyncio

import pytest

import alfa1_agent
import alfa1_tools
from alfa1_agent import _describe_tool_attempt, _extract_file_writes, run_agent_turn


@pytest.fixture(autouse=True)
def _redirect_last_workspace_pointer(tmp_path_factory, monkeypatch):
    # Several tests below call alfa1_tools.set_workspace() directly, which
    # now also persists a "last used workspace" pointer file — redirect it
    # so tests never read/write the real user's remembered workspace (see
    # test_alfa1_tools.py's equivalent fixture for the full reasoning).
    monkeypatch.setattr(
        alfa1_tools, "_LAST_WORKSPACE_PATH",
        tmp_path_factory.mktemp("alfa1_last_ws") / ".alfa1_last_workspace.json",
    )


def test_extract_file_writes_finds_backtick_header_before_fence():
    content = (
        "Here is the file:\n\n"
        "`src/app.py`\n"
        "```python\n"
        "print('hi')\n"
        "```\n"
    )
    writes = _extract_file_writes(content)
    assert writes == [("src/app.py", "print('hi')\n")]


def test_extract_file_writes_ignores_fence_with_no_filename_header():
    content = "Some text\n```python\nprint('hi')\n```\n"
    assert _extract_file_writes(content) == []


def test_extract_file_writes_finds_filename_comment_inside_fence():
    """Matches the shape patch_engine.reconstruct_full_file_response actually
    produces after applying a SEARCH/REPLACE patch: no header line before the
    fence, just a `# path` comment as the fence's first line — this must be
    recognized or every patch-mode edit would be silently dropped."""
    content = (
        "[ShapeShifter: 1 patch applied] — complete updated file:\n\n"
        "```python\n"
        "# calc.py\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def multiply(a, b):\n"
        "    return a * b\n"
        "```\n"
    )
    writes = _extract_file_writes(content)
    assert len(writes) == 1
    path, written = writes[0]
    assert path == "calc.py"
    assert "# calc.py" not in written
    assert "def multiply(a, b):" in written
    assert "def add(a, b):" in written


def test_extract_file_writes_handles_malformed_intro_before_reconstruction():
    """Regression test for a real observed bug: the model started its own
    `path`/```lang file-write convention, then abandoned it mid-way to emit
    SEARCH/REPLACE patch markers instead — leaving an odd number of ``` in
    its own prose. ShapeShifter's patch reconstruction then appended its own
    status line + a fresh fence with the complete file. Non-greedy generic
    fence-pairing latched onto the wrong two ``` markers and extracted only
    a ~60-byte fragment of the status line as "the file", silently
    truncating real content. This is the exact content captured from that
    session (only the code body's length is trimmed here for brevity)."""
    content = (
        "Aggiungo il supporto per la radice quadrata. Modifico `calculator.py` "
        "per gestire anche espressioni unarie (`sqrt 25`).\n\n"
        "`calculator.py`\n"
        "```python\n"
        "\n"
        "[ShapeShifter: 3 patches applied] — complete updated file:\n\n"
        "```python\n"
        "# calculator.py\n"
        "def calculator():\n"
        "    print('hi')\n"
        "\n"
        "def sqrt(a):\n"
        "    return a ** 0.5\n"
        "```"
    )
    writes = _extract_file_writes(content)
    assert len(writes) == 1
    path, written = writes[0]
    assert path == "calculator.py"
    assert "# calculator.py" not in written
    assert "def calculator():" in written
    assert "def sqrt(a):" in written
    assert "return a ** 0.5" in written
    # the truncated-to-status-line bug this guards against:
    assert "ShapeShifter" not in written


def test_extract_file_writes_rejects_reconstruction_with_leftover_patch_markers():
    """Regression test for a real observed bug: the model imitated
    ShapeShifter's own "[ShapeShifter: N patches applied]" status line
    itself (likely echoing a pattern it had seen earlier in its own
    conversation history) while ALSO tacking on a second, genuinely
    unresolved SEARCH/REPLACE attempt right after it. Because
    _SHAPESHIFTER_RECONSTRUCTED_RE matches greedily to the end of the
    string, that trailing raw patch got swallowed into "body" and written
    to disk verbatim, <<<<<<< SEARCH markers and all — this must be
    rejected instead so run_agent_turn's _try_apply_raw_patch handles the
    real patch op properly against the actual file."""
    content = (
        "`index.html`\n```html\n\n"
        "[ShapeShifter: 2 patches applied] — complete updated file:\n\n"
        "```html\n"
        "# index.html\n"
        "<!DOCTYPE html>\n"
        "<html></html>\n"
        ">>>>>>> REPLACE\n"  # stray, unmatched closing marker — no real opener before it
        "```\n\n"
        "`index.html`\n```html\n"
        "<<<<<<< SEARCH\n"
        "        function clearDisplay() {\n"
        "=======\n"
        "        function sqrt() { return Math.sqrt(x); }\n\n"
        "        function clearDisplay() {\n"
        ">>>>>>> REPLACE\n"
        "```"
    )
    assert _extract_file_writes(content) == []


def test_run_agent_turn_applies_the_real_patch_when_reconstruction_has_leftover_markers(monkeypatch, tmp_path):
    """End-to-end version of the above: the turn must still make correct
    progress (apply the one well-formed patch against the real file) rather
    than silently doing nothing, now that the untrustworthy "reconstruction"
    is rejected."""
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "index.html").write_text(
        "<html>\n<script>\n        function clearDisplay() {\n            x = 0;\n        }\n</script>\n</html>\n",
        encoding="utf-8",
    )

    calls = {"n": 0}

    async def fake_call_self(conversation, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message(
                "`index.html`\n```html\n\n"
                "[ShapeShifter: 1 patch applied] — complete updated file:\n\n"
                "```html\n"
                "# index.html\n"
                "<html></html>\n"
                ">>>>>>> REPLACE\n"
                "```\n\n"
                "`index.html`\n```html\n"
                "<<<<<<< SEARCH\n"
                "        function clearDisplay() {\n"
                "=======\n"
                "        function sqrt() { return 1; }\n\n"
                "        function clearDisplay() {\n"
                ">>>>>>> REPLACE\n"
                "```"
            )
        return _assistant_message("done")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    conversation = [{"role": "user", "content": "add sqrt"}]
    asyncio.run(run_agent_turn(conversation))

    written = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "<<<<<<< SEARCH" not in written
    assert ">>>>>>> REPLACE" not in written
    assert "function sqrt()" in written
    assert "function clearDisplay()" in written


def test_extract_file_writes_reconstruction_without_filename_comment_falls_back_to_intro_header():
    """If the artifact key was a __lang__ fallback (no real filename known to
    ShapeShifter yet), patch_engine emits no `# path` comment inside the
    fence — the model's own backtick header before its (possibly malformed)
    intro is the only remaining source for the path."""
    content = (
        "`notes.md`\n"
        "[ShapeShifter: 1 patch applied] — complete updated file:\n\n"
        "```md\n"
        "# Notes\n"
        "updated content\n"
        "```"
    )
    writes = _extract_file_writes(content)
    assert writes == [("notes.md", "# Notes\nupdated content\n")]


def test_extract_file_writes_finds_multiple_files():
    content = (
        "`a.py`\n```python\nx = 1\n```\n\n"
        "`b.py`\n```python\ny = 2\n```\n"
    )
    writes = _extract_file_writes(content)
    assert writes == [("a.py", "x = 1\n"), ("b.py", "y = 2\n")]


def _assistant_message(content, finish_reason="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}]}


def test_run_agent_turn_no_actions_returns_immediately(monkeypatch):
    events = []

    async def fake_call_self(conversation, model):
        return _assistant_message("hello there")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "hi"}]
    asyncio.run(run_agent_turn(conversation, on_event=on_event))

    assert conversation[-1]["content"] == "hello there"
    assert events == [{"type": "thinking"}, {"type": "assistant", "content": "hello there"}]
    # a system prompt should have been injected at the front
    assert conversation[0]["role"] == "system"


def test_run_agent_turn_writes_file_then_stops(monkeypatch, tmp_path):
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))

    calls = {"n": 0}

    async def fake_call_self(conversation, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message("`a.txt`\n```\nhello\n```\n")
        return _assistant_message("done")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "write a file"}]
    asyncio.run(run_agent_turn(conversation, on_event=on_event))

    assert (tmp_path / "a.txt").read_text() == "hello\n"
    event_types = [e["type"] for e in events]
    assert event_types == ["thinking", "tool_call", "tool_result", "thinking", "assistant"]
    assert conversation[-1]["content"] == "done"
    # a synthetic user message reporting the outcome should have been appended
    user_msgs = [m for m in conversation if m.get("role") == "user"]
    assert any("Wrote a.txt" in m["content"] for m in user_msgs)


def test_run_agent_turn_executes_run_command_tag(monkeypatch, tmp_path):
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))
    import sys

    calls = {"n": 0}

    async def fake_call_self(conversation, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message(
                f'<alfa1:run_command>"{sys.executable}" -c "print(42)"</alfa1:run_command>'
            )
        return _assistant_message("saw the output")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    conversation = [{"role": "user", "content": "run something"}]
    asyncio.run(run_agent_turn(conversation))

    user_msgs = [m for m in conversation if m.get("role") == "user"]
    assert any("42" in m["content"] for m in user_msgs)
    assert conversation[-1]["content"] == "saw the output"


def test_run_agent_turn_read_file_wraps_result_as_artifact(monkeypatch, tmp_path):
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "existing.py").write_text("x = 1\n")

    calls = {"n": 0}

    async def fake_call_self(conversation, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message("<alfa1:read_file>existing.py</alfa1:read_file>")
        return _assistant_message("ok")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    conversation = [{"role": "user", "content": "read a file"}]
    asyncio.run(run_agent_turn(conversation))

    user_msgs = [m for m in conversation if m.get("role") == "user"]
    combined = "\n".join(m["content"] for m in user_msgs)
    assert "`existing.py`" in combined
    assert "x = 1" in combined


def test_run_agent_turn_search_files_action(monkeypatch, tmp_path):
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "a.py").write_text("def target_fn():\n    pass\n", encoding="utf-8")

    calls = {"n": 0}

    async def fake_call_self(conversation, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message("<alfa1:search_files>target_fn</alfa1:search_files>")
        return _assistant_message("found it")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    conversation = [{"role": "user", "content": "find target_fn"}]
    asyncio.run(run_agent_turn(conversation))

    user_msgs = [m for m in conversation if m.get("role") == "user"]
    combined = "\n".join(m["content"] for m in user_msgs)
    assert "a.py:1" in combined
    assert conversation[-1]["content"] == "found it"


def test_run_agent_turn_action_error_does_not_crash_loop(monkeypatch, tmp_path):
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))

    calls = {"n": 0}

    async def fake_call_self(conversation, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message("<alfa1:read_file>missing.txt</alfa1:read_file>")
        return _assistant_message("handled the error")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    conversation = [{"role": "user", "content": "read a missing file"}]
    asyncio.run(run_agent_turn(conversation))

    user_msgs = [m for m in conversation if m.get("role") == "user"]
    assert any("Error" in m["content"] for m in user_msgs)
    assert conversation[-1]["content"] == "handled the error"


def test_extract_file_writes_does_not_write_unresolved_raw_patch_markers(tmp_path):
    """Regression test for a real observed bug: when ShapeShifter's own
    patch reconstruction fails to resolve a target artifact (e.g. its
    in-memory view drifted from the real file), it passes the model's raw
    SEARCH/REPLACE text through untouched. The generic file-write scan must
    NOT treat that raw patch text as literal file content — _extract_file_writes
    should find nothing here so run_agent_turn's dedicated patch-application
    path (_try_apply_raw_patch) handles it instead."""
    content = (
        "`calculator.py`\n```python\n"
        "<<<<<<< SEARCH\n"
        '    print("a")\n'
        "=======\n"
        '    print("b")\n'
        ">>>>>>> REPLACE\n"
        "```"
    )
    assert _extract_file_writes(content) == []


def test_run_agent_turn_applies_raw_unresolved_patch_directly_to_disk(monkeypatch, tmp_path):
    """The core regression: previously this raw patch text got written to
    calculator.py VERBATIM (literally the <<<<<<< SEARCH markers), corrupting
    the file. Now Alfa1 applies it itself against the real current content."""
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "calculator.py").write_text(
        'def calculator():\n    print("Operazioni disponibili: +, -, *, /")\n', encoding="utf-8",
    )

    calls = {"n": 0}

    async def fake_call_self(conversation, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message(
                "`calculator.py`\n```python\n"
                "<<<<<<< SEARCH\n"
                '    print("Operazioni disponibili: +, -, *, /")\n'
                "=======\n"
                '    print("Operazioni disponibili: +, -, *, /, ^")\n'
                ">>>>>>> REPLACE\n"
                "```"
            )
        return _assistant_message("done")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "add a caret operator"}]
    asyncio.run(run_agent_turn(conversation, on_event=on_event))

    written = (tmp_path / "calculator.py").read_text(encoding="utf-8")
    assert "<<<<<<< SEARCH" not in written
    assert "Operazioni disponibili: +, -, *, /, ^" in written
    assert "def calculator():" in written

    tool_events = [e for e in events if e["type"] in ("tool_call", "tool_result")]
    assert tool_events[0]["name"] == "apply_patch"
    assert "1 patch applied" in tool_events[1]["result"]


def test_run_agent_turn_raw_patch_that_matches_nothing_reports_current_content(monkeypatch, tmp_path):
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "calculator.py").write_text("def calculator():\n    pass\n", encoding="utf-8")

    calls = {"n": 0}

    async def fake_call_self(conversation, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message(
                "`calculator.py`\n```python\n"
                "<<<<<<< SEARCH\n"
                "this text does not exist in the file\n"
                "=======\n"
                "replacement\n"
                ">>>>>>> REPLACE\n"
                "```"
            )
        return _assistant_message("retrying with correct content")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    conversation = [{"role": "user", "content": "edit it"}]
    asyncio.run(run_agent_turn(conversation))

    # file must be left untouched, not corrupted with the failed patch text
    assert (tmp_path / "calculator.py").read_text(encoding="utf-8") == "def calculator():\n    pass\n"
    user_msgs = [m for m in conversation if m.get("role") == "user"]
    assert any("did not match" in m["content"] or "None of the" in m["content"] for m in user_msgs)


def _assistant_message_with_reasoning(content, reasoning):
    return {"choices": [{"message": {"role": "assistant", "content": content, "reasoning": reasoning}}]}


def test_run_agent_turn_emits_reasoning_event(monkeypatch):
    async def fake_call_self(conversation, model):
        return _assistant_message_with_reasoning("hello there", "thinking about it")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "hi"}]
    asyncio.run(run_agent_turn(conversation, on_event=on_event))

    assert {"type": "reasoning", "content": "thinking about it"} in events


def test_describe_tool_attempt_from_structured_native_tool_calls():
    tool_calls = [{"function": {"name": "shapeshifter_read_file", "arguments": '{"path": "server.py"}'}}]
    desc = _describe_tool_attempt(tool_calls, "")
    assert "shapeshifter_read_file" in desc
    assert "server.py" in desc


def test_describe_tool_attempt_from_dsml_style_text():
    content = (
        '<|DSML|tool_calls> <|DSML|invoke name="read_file"> '
        '<|DSML|parameter name="path" string="true">minismtp.py</|DSML|parameter> '
        '</|DSML|invoke> </|DSML|tool_calls>'
    )
    desc = _describe_tool_attempt(None, content)
    assert "read_file" in desc
    assert "minismtp.py" in desc


def test_describe_tool_attempt_falls_back_to_raw_snippet():
    desc = _describe_tool_attempt(None, "some garbled text with no recognizable pattern")
    assert "garbled text" in desc


def test_describe_tool_attempt_handles_empty_reply():
    assert "empty" in _describe_tool_attempt(None, "").lower()


def test_run_agent_turn_includes_description_in_unrecognized_event(monkeypatch):
    async def fake_call_self(conversation, model):
        return _assistant_message_with_tool_calls(
            "", [{"function": {"name": "shapeshifter_read_file", "arguments": '{"path": "a.py"}'}}],
        ) if fake_call_self.calls == 0 else _assistant_message("ok")

    fake_call_self.calls = 0

    async def wrapped(conversation, model):
        result = await fake_call_self(conversation, model)
        fake_call_self.calls += 1
        return result

    monkeypatch.setattr(alfa1_agent, "_call_self", wrapped)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "read a.py"}]
    asyncio.run(run_agent_turn(conversation, on_event=on_event))

    attempt_events = [e for e in events if e["type"] == "tool_attempt_unrecognized"]
    assert len(attempt_events) == 1
    assert "shapeshifter_read_file" in attempt_events[0]["description"]
    assert "a.py" in attempt_events[0]["description"]


def test_run_agent_turn_detects_unrecognized_tool_syntax_and_retries(monkeypatch):
    calls = {"n": 0}

    async def fake_call_self(conversation, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message(
                '<|DSML|tool_calls> <|DSML|invoke name="read_file"> '
                '<|DSML|parameter name="path" string="true">x.py</|DSML|parameter> '
                '</|DSML|invoke> </|DSML|tool_calls>'
            )
        return _assistant_message("ok, using the right format now")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "read a file"}]
    asyncio.run(run_agent_turn(conversation, on_event=on_event))

    assert any(e["type"] == "tool_attempt_unrecognized" for e in events)
    assert calls["n"] == 2
    assert conversation[-1]["content"] == "ok, using the right format now"
    corrective = [m for m in conversation if m.get("role") == "user" and "unsupported tool-calling" in m.get("content", "")]
    assert len(corrective) == 1


def _assistant_message_with_tool_calls(content, tool_calls):
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}]}


def test_run_agent_turn_strips_native_tool_calls_before_storing_in_history(monkeypatch):
    """Regression test for a real observed incident: deepseek-v4-flash (via
    OpenRouter) emitted a native tool_calls structure (`shapeshifter_read_file`)
    even though this request never declared any `tools`. Left in
    `conversation`, that field makes wrapper_server's _is_agentic() treat
    every LATER request in the session as agentic passthrough too — silently
    disabling both input compression and output patch savings for the rest
    of the conversation, with no visible error. It must never survive into
    stored history."""
    calls = {"n": 0}

    async def fake_call_self(conversation, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message_with_tool_calls(
                "", [{"type": "function", "id": "call_1",
                      "function": {"name": "shapeshifter_read_file", "arguments": '{"path": "server.py"}'}}],
            )
        return _assistant_message("ok, using the right format now")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    conversation = [{"role": "user", "content": "read a file"}]
    asyncio.run(run_agent_turn(conversation))

    stored_assistant_msgs = [m for m in conversation if m.get("role") == "assistant"]
    assert all("tool_calls" not in m for m in stored_assistant_msgs)
    assert all("function_call" not in m for m in stored_assistant_msgs)


def test_run_agent_turn_treats_native_tool_call_attempt_as_unrecognized(monkeypatch):
    calls = {"n": 0}

    async def fake_call_self(conversation, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message_with_tool_calls(
                "", [{"type": "function", "id": "call_1",
                      "function": {"name": "shapeshifter_read_file", "arguments": '{"path": "server.py"}'}}],
            )
        return _assistant_message("ok, using the right format now")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "read a file"}]
    asyncio.run(run_agent_turn(conversation, on_event=on_event))

    assert any(e["type"] == "tool_attempt_unrecognized" for e in events)
    assert calls["n"] == 2
    assert conversation[-1]["content"] == "ok, using the right format now"
    corrective = [m for m in conversation if m.get("role") == "user" and "unsupported tool-calling" in m.get("content", "")]
    assert len(corrective) == 1


def test_run_agent_turn_detects_truncated_reply_and_retries(monkeypatch):
    """Regression test for a real observed bug: a verbose reasoning model's
    response got cut off by the output length limit mid-action-tag (observed:
    `<alfa1:read_file>calc` with no closing `.py</alfa1:read_file>`). A
    truncated tag matches neither _ACTION_TAG_RE nor a file write, so without
    checking finish_reason=="length" this would have silently become the
    turn's final answer instead of retrying."""
    calls = {"n": 0}

    async def fake_call_self(conversation, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message("<alfa1:read_file>calc", finish_reason="length")
        return _assistant_message("done", finish_reason="stop")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "add sqrt"}]
    asyncio.run(run_agent_turn(conversation, on_event=on_event))

    assert any(e["type"] == "truncated" for e in events)
    assert calls["n"] == 2
    assert conversation[-1]["content"] == "done"
    corrective = [m for m in conversation if m.get("role") == "user" and "cut off" in m.get("content", "")]
    assert len(corrective) == 1


def test_call_self_requests_a_generous_max_tokens(monkeypatch):
    """Regression guard for the same truncation bug from the other
    direction: the loopback call must ask for enough tokens that a verbose
    reasoning model's response isn't cut off mid-action in the first place —
    the proxy's own DEFAULT_MAX_OUTPUT_TOKENS (1200) was observed to be too
    small for this in practice."""
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return _assistant_message("ok")

    class FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json):
            captured.update(json)
            return FakeResponse()

    monkeypatch.setattr(alfa1_agent.httpx, "AsyncClient", FakeAsyncClient)

    asyncio.run(alfa1_agent._call_self([{"role": "user", "content": "hi"}], None))
    assert captured["max_tokens"] >= 4096


def test_run_agent_turn_stops_at_max_iterations(monkeypatch, tmp_path):
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))

    async def fake_call_self(conversation, model):
        return _assistant_message("<alfa1:list_files>.</alfa1:list_files>")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "loop forever"}]
    asyncio.run(run_agent_turn(conversation, max_iterations=2, on_event=on_event))

    assert events[-1]["type"] == "assistant"
    assert "stopped" in events[-1]["content"] or events[-1]["content"]
