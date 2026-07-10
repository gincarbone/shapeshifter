"""Tests for alfa1_agent.py — the plain-text action protocol (file writes via
fenced code blocks with a filename header, and <alfa1:...> action tags) and
the agent loop. No OpenAI tool-calling is used here on purpose: see the
module docstring in alfa1_agent.py for why (it would make wrapper_server's
_is_agentic bypass ShapeShifter's own input/output token-savings pipeline)."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import alfa1_agent
import alfa1_tools
from alfa1_agent import (
    _describe_tool_attempt, _extract_file_writes, _resolve_patch_target_path,
    _try_apply_raw_patch, _verify_content, run_agent_turn,
)


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

    async def fake_call_self(conversation, model, on_delta=None):
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

    async def fake_call_self(conversation, model, on_delta=None):
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


def test_verify_content_accepts_valid_python():
    assert _verify_content("app.py", "def f():\n    return 1\n") is None


def test_verify_content_rejects_syntax_broken_python():
    problem = _verify_content("app.py", "def f(:\n    return 1\n")
    assert problem is not None
    assert "syntax error" in problem


def test_verify_content_rejects_leftover_patch_markers():
    problem = _verify_content("app.py", "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n")
    assert problem is not None
    assert "patch markers" in problem


def test_verify_content_accepts_valid_json():
    assert _verify_content("data.json", '{"a": 1}') is None


def test_verify_content_rejects_broken_json():
    problem = _verify_content("data.json", '{"a": 1,}')
    assert problem is not None
    assert "invalid" in problem


def test_verify_content_ignores_non_checked_extensions():
    # .txt/.md/etc. have no dedicated syntax check — anything not caught by
    # the universal patch-marker check is accepted.
    assert _verify_content("notes.txt", "this is not valid python at all (") is None


def test_run_agent_turn_refuses_to_write_syntax_broken_python(monkeypatch, tmp_path):
    """The verifier must reject bad content BEFORE it touches disk — the
    file must be left exactly as it was, not corrupted, and the model gets
    a corrective message instead of a silently broken file."""
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))

    calls = {"n": 0}

    async def fake_call_self(conversation, model, on_delta=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message("`app.py`\n```python\ndef f(:\n    return 1\n```\n")
        return _assistant_message("done")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "write app.py"}]
    asyncio.run(run_agent_turn(conversation, on_event=on_event))

    assert not (tmp_path / "app.py").exists()
    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert tool_results and "Refused to write" in tool_results[0]["result"]
    assert "syntax error" in tool_results[0]["result"]


def test_run_agent_turn_writes_file_then_stops(monkeypatch, tmp_path):
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))

    calls = {"n": 0}

    async def fake_call_self(conversation, model, on_delta=None):
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

    async def fake_call_self(conversation, model, on_delta=None):
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

    async def fake_call_self(conversation, model, on_delta=None):
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

    async def fake_call_self(conversation, model, on_delta=None):
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

    async def fake_call_self(conversation, model, on_delta=None):
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


def test_resolve_patch_target_path_finds_header_before_the_fence_not_the_marker():
    """Regression test for a real observed bug: every apply_patch attempt in
    a live session failed repeatedly with "didn't clearly specify which
    file it targets" — {"path": null} — even though the model correctly
    wrote `calculator.py` right before the fence every single time. The
    marker sits INSIDE the fence, several lines after the header, so
    searching for the header immediately before the marker's position
    always missed it; the fix looks right before the fence-open instead."""
    content = (
        "`calculator.py`\n```python\n"
        "<<<<<<< SEARCH\n"
        "        elif op == \"/\":\n"
        "=======\n"
        "        elif op == \"**\":\n"
        ">>>>>>> REPLACE\n"
        "```"
    )
    marker_pos = content.index("<<<<<<< SEARCH")
    assert _resolve_patch_target_path(content, marker_pos) == "calculator.py"


def test_try_apply_raw_patch_resolves_path_with_header_before_fence(tmp_path):
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "calculator.py").write_text(
        'def calculator(op):\n    if op == "/":\n        pass\n', encoding="utf-8",
    )
    content = (
        "`calculator.py`\n```python\n"
        "<<<<<<< SEARCH\n"
        '    if op == "/":\n'
        "=======\n"
        '    if op == "**":\n'
        ">>>>>>> REPLACE\n"
        "```"
    )
    result = _try_apply_raw_patch(content)
    assert result is not None
    path, outcome, new_full_content = result
    assert path == "calculator.py"
    assert "1 patch applied" in outcome
    assert new_full_content is not None
    assert 'if op == "**":' in new_full_content
    assert 'if op == "**":' in (tmp_path / "calculator.py").read_text(encoding="utf-8")


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

    async def fake_call_self(conversation, model, on_delta=None):
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

    async def fake_call_self(conversation, model, on_delta=None):
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
    # Regression: observed in practice — after a failed patch match (one
    # missing line vs. the real file), the model fell back to rewriting the
    # whole file instead of retrying with a corrected patch, silently
    # forfeiting output-token savings for that turn even though nothing
    # forced it to. The retry message must explicitly steer it back to
    # patching instead of just showing the content and hoping.
    assert any("do NOT rewrite the whole file" in m["content"] for m in user_msgs)


def test_run_agent_turn_resolves_patch_path_with_multiple_files_in_workspace(monkeypatch, tmp_path):
    """The single-file-in-workspace fallback in _resolve_patch_target_path
    masked the header-adjacency bug in earlier tests (a tmp_path with only
    one file always resolved correctly "by luck" through the fallback, even
    while the primary header-detection path was broken). The real session
    that surfaced the bug had multiple files, where the fallback can't
    apply and correct header detection is the only way to resolve the
    target — repro that shape here so the fallback can't mask a regression."""
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "calculator.py").write_text(
        'def calculator(op):\n    if op == "/":\n        pass\n', encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("notes\n", encoding="utf-8")

    calls = {"n": 0}

    async def fake_call_self(conversation, model, on_delta=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _assistant_message(
                "`calculator.py`\n```python\n"
                "<<<<<<< SEARCH\n"
                '    if op == "/":\n'
                "=======\n"
                '    if op == "**":\n'
                ">>>>>>> REPLACE\n"
                "```"
            )
        return _assistant_message("done")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "add power operator"}]
    asyncio.run(run_agent_turn(conversation, on_event=on_event))

    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert tool_results and "1 patch applied" in tool_results[0]["result"]
    assert 'if op == "**":' in (tmp_path / "calculator.py").read_text(encoding="utf-8")


def test_run_agent_turn_rewrites_history_after_successful_patch_to_avoid_stale_context(monkeypatch, tmp_path):
    """Regression test for a real observed bug: after a successful
    apply_patch, the stored assistant message still contained the model's
    raw <<<<<<< SEARCH text (the file on disk was correctly patched, but the
    CONVERSATION HISTORY was not updated to match). Since transformers.py's
    own artifact tracking (_extract_latest_artifacts) scans conversation
    TEXT for the latest code block per file — with no idea this module
    silently patched the real file out-of-band — every later turn kept
    being shown stale patch syntax as "the current file", and the model
    visibly got confused about which byte count/version was current,
    flip-flopping across several turns instead of making progress. The
    stored message must be rewritten to show the real resulting content."""
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "calculator.py").write_text(
        'def calculator(op):\n    if op == "/":\n        pass\n', encoding="utf-8",
    )

    async def fake_call_self(conversation, model, on_delta=None):
        return _assistant_message(
            "`calculator.py`\n```python\n"
            "<<<<<<< SEARCH\n"
            '    if op == "/":\n'
            "=======\n"
            '    if op == "**":\n'
            ">>>>>>> REPLACE\n"
            "```"
        )

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    conversation = [{"role": "user", "content": "add power operator"}]
    asyncio.run(run_agent_turn(conversation, max_iterations=1))

    assistant_msgs = [m for m in conversation if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    stored_content = assistant_msgs[0]["content"]
    assert "<<<<<<< SEARCH" not in stored_content
    assert ">>>>>>> REPLACE" not in stored_content
    assert 'if op == "**":' in stored_content
    assert "`calculator.py`" in stored_content


def _assistant_message_with_reasoning(content, reasoning):
    return {"choices": [{"message": {"role": "assistant", "content": content, "reasoning": reasoning}}]}


def test_run_agent_turn_emits_reasoning_event(monkeypatch):
    async def fake_call_self(conversation, model, on_delta=None):
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
    async def fake_call_self(conversation, model, on_delta=None):
        return _assistant_message_with_tool_calls(
            "", [{"function": {"name": "shapeshifter_read_file", "arguments": '{"path": "a.py"}'}}],
        ) if fake_call_self.calls == 0 else _assistant_message("ok")

    fake_call_self.calls = 0

    async def wrapped(conversation, model, on_delta=None):
        result = await fake_call_self(conversation, model, on_delta)
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

    async def fake_call_self(conversation, model, on_delta=None):
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

    async def fake_call_self(conversation, model, on_delta=None):
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

    async def fake_call_self(conversation, model, on_delta=None):
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

    async def fake_call_self(conversation, model, on_delta=None):
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


def _sse_transport(body: bytes, captured_requests: list | None = None):
    """httpx.MockTransport serving a canned SSE stream — same technique
    proven in tests/test_llm_client.py's real-streaming tests, reused here
    since _call_self now streams from the proxy the same way llm_client's
    stream_upstream streams from the upstream provider."""
    def handler(request):
        if captured_requests is not None:
            captured_requests.append(json.loads(request.content))
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
    return httpx.MockTransport(handler)


def test_call_self_requests_a_generous_max_tokens(monkeypatch):
    """Regression guard for the same truncation bug from the other
    direction: the loopback call must ask for enough tokens that a verbose
    reasoning model's response isn't cut off mid-action in the first place —
    the proxy's own DEFAULT_MAX_OUTPUT_TOKENS (1200) was observed to be too
    small for this in practice."""
    body = (
        b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
        b'data: [DONE]\n\n'
    )
    captured = []
    real_async_client = httpx.AsyncClient

    def factory(*a, **kw):
        return real_async_client(transport=_sse_transport(body, captured))
    monkeypatch.setattr(alfa1_agent.httpx, "AsyncClient", factory)

    asyncio.run(alfa1_agent._call_self([{"role": "user", "content": "hi"}], None))
    assert captured[0]["max_tokens"] >= 4096
    assert captured[0]["stream"] is True


def test_call_self_parses_streamed_chunks_and_invokes_on_delta(monkeypatch):
    body = (
        b'data: {"choices":[{"index":0,"delta":{"reasoning":"thinking..."}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}\n\n'
        b'data: [DONE]\n\n'
    )
    real_async_client = httpx.AsyncClient

    def factory(*a, **kw):
        return real_async_client(transport=_sse_transport(body))
    monkeypatch.setattr(alfa1_agent.httpx, "AsyncClient", factory)

    deltas = []

    async def on_delta(kind, text):
        deltas.append((kind, text))

    resp = asyncio.run(alfa1_agent._call_self([{"role": "user", "content": "hi"}], None, on_delta=on_delta))
    assert deltas == [("reasoning", "thinking..."), ("content", "Hello"), ("content", " world")]
    message = resp["choices"][0]["message"]
    assert message["content"] == "Hello world"
    assert message["reasoning"] == "thinking..."
    assert resp["choices"][0]["finish_reason"] == "stop"


def test_call_self_reassembles_streamed_tool_call_fragments(monkeypatch):
    chunks = [
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "read_", "arguments": ""}},
        ]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"name": "file", "arguments": '{"path"'}},
        ]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ': "a.py"}'}},
        ]}}]},
    ]
    body = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"
    real_async_client = httpx.AsyncClient

    def factory(*a, **kw):
        return real_async_client(transport=_sse_transport(body))
    monkeypatch.setattr(alfa1_agent.httpx, "AsyncClient", factory)

    resp = asyncio.run(alfa1_agent._call_self([{"role": "user", "content": "hi"}], None))
    tool_calls = resp["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_1"
    assert tool_calls[0]["function"]["name"] == "read_file"
    assert tool_calls[0]["function"]["arguments"] == '{"path": "a.py"}'


def test_run_agent_turn_stops_at_max_iterations(monkeypatch, tmp_path):
    import alfa1_tools
    alfa1_tools.set_workspace(str(tmp_path))

    async def fake_call_self(conversation, model, on_delta=None):
        return _assistant_message("<alfa1:list_files>.</alfa1:list_files>")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "loop forever"}]
    result = asyncio.run(run_agent_turn(conversation, max_iterations=2, on_event=on_event))

    assert result == "paused"
    assert events[-1]["type"] == "iteration_cap_reached"
    assert events[-1]["max_iterations"] == 2


def test_tdd_gate_does_not_run_tests_when_not_configured(monkeypatch, tmp_path):
    """Default is disabled (see alfa1_tools._DEFAULT_TDD_CONFIG) — a file
    write must not trigger any test run unless the workspace opted in."""
    alfa1_tools.set_workspace(str(tmp_path))

    # First call writes the file (loop continues); second call has no more
    # actions, so a 2nd iteration is needed to reach the "done" check.
    replies = iter([
        _assistant_message("`app.py`\n```python\nprint('hi')\n```"),
        _assistant_message("Done."),
    ])

    async def fake_call_self(conversation, model, on_delta=None):
        return next(replies)

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    async def fail_if_called(*a, **k):
        raise AssertionError("run_command must not be called when TDD is disabled")

    monkeypatch.setattr(alfa1_tools, "run_command", fail_if_called)

    conversation = [{"role": "user", "content": "add a print"}]
    result = asyncio.run(run_agent_turn(conversation))
    assert result == "done"


def test_tdd_gate_forces_a_retry_on_failure_then_lets_the_turn_end_on_pass(monkeypatch, tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    alfa1_tools.save_tdd_config({"enabled": True, "test_command": "pytest", "max_retries": 5})

    replies = iter([
        _assistant_message("`app.py`\n```python\nprint('hi')\n```"),  # writes a file
        _assistant_message("Should be fixed now."),                    # claims done -> forced test -> FAILS
        _assistant_message("Fixed for real."),                          # claims done again -> forced test -> PASSES
    ])

    async def fake_call_self(conversation, model, on_delta=None):
        return next(replies)

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    test_runs = iter([
        {"exit_code": 1, "stdout": "1 failed", "stderr": "", "timed_out": False, "truncated": False, "duration_ms": 1.0},
        {"exit_code": 0, "stdout": "5 passed", "stderr": "", "timed_out": False, "truncated": False, "duration_ms": 1.0},
    ])

    async def fake_run_command(command, *a, **k):
        assert command == "pytest"
        return next(test_runs)

    monkeypatch.setattr(alfa1_tools, "run_command", fake_run_command)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "add a print"}]
    result = asyncio.run(run_agent_turn(conversation, on_event=on_event))

    assert result == "done"
    tdd_results = [e for e in events if e["type"] == "tdd_result"]
    assert [e["passed"] for e in tdd_results] == [False, True]
    assert any(
        m.get("role") == "user" and "FAILED" in m.get("content", "")
        for m in conversation
    )


def test_tdd_gate_pauses_after_exhausting_retries(monkeypatch, tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    alfa1_tools.save_tdd_config({"enabled": True, "test_command": "pytest", "max_retries": 2})

    call_n = {"n": 0}

    async def fake_call_self(conversation, model, on_delta=None):
        call_n["n"] += 1
        if call_n["n"] == 1:
            return _assistant_message("`app.py`\n```python\nprint('hi')\n```")
        return _assistant_message("Should be fixed now.")

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    async def fake_run_command(command, *a, **k):
        return {"exit_code": 1, "stdout": "still failing", "stderr": "", "timed_out": False,
                "truncated": False, "duration_ms": 1.0}

    monkeypatch.setattr(alfa1_tools, "run_command", fake_run_command)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "add a print"}]
    result = asyncio.run(run_agent_turn(conversation, on_event=on_event, max_iterations=10))

    assert result == "paused"
    assert events[-1]["type"] == "tdd_retries_exhausted"
    assert events[-1]["max_retries"] == 2


def test_tdd_gate_surfaces_a_broken_test_command_as_an_error(monkeypatch, tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    alfa1_tools.save_tdd_config({"enabled": True, "test_command": "pytest", "max_retries": 5})

    replies = iter([
        _assistant_message("`app.py`\n```python\nprint('hi')\n```"),
        _assistant_message("Done."),
    ])

    async def fake_call_self(conversation, model, on_delta=None):
        return next(replies)

    monkeypatch.setattr(alfa1_agent, "_call_self", fake_call_self)

    async def fake_run_command(command, *a, **k):
        raise alfa1_tools.Alfa1Error("cwd is not a directory")

    monkeypatch.setattr(alfa1_tools, "run_command", fake_run_command)

    events = []

    async def on_event(evt):
        events.append(evt)

    conversation = [{"role": "user", "content": "add a print"}]
    result = asyncio.run(run_agent_turn(conversation, on_event=on_event))

    assert result == "error"
    assert events[-1]["type"] == "error"
