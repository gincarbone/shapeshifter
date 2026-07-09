# Copyright (c) 2026 Gaetano Marcello Incarbone. MIT License — see LICENSE file.
"""Alfa1 — agentic loop.

Talks to ShapeShifter's OWN /v1/chat/completions endpoint over HTTP loopback,
using a PLAIN-TEXT action protocol instead of OpenAI native tool-calling.

This is deliberate, not a simplification for its own sake: wrapper_server's
`_is_agentic` (wrapper_server.py:339-359) treats ANY request that declares a
`tools`/`tool_choice` field, or contains `tool_calls`/`role: tool` messages,
as an agentic tool-calling exchange and routes it through raw passthrough —
skipping BOTH the input-side context compression (transformers.py) and the
output-side patch-reconstruction pipeline (patch_engine.py, Features O1-O4
in docs/token_savings_roadmap.md) entirely. Those savings are the entire
point of ShapeShifter and of Alfa1 using it, so Alfa1's agent must look like
an ordinary prompt-based coding client (Cline/Aider-style) to the proxy:
plain chat messages, file edits expressed as fenced code blocks with a
filename header, no `tools` field ever set on the request body.

File-write convention (matches transformers.py's own artifact-detection
heuristics exactly, so ShapeShifter's coding-session/patch-mode detection
recognizes Alfa1's own turns the same way it would recognize Cline's):
on its own line immediately before a fenced code block, the file path
wrapped in backticks, e.g.:

    `src/app.py`
    ```python
    ...full file content...
    ```

From the second turn touching a given file onward, ShapeShifter injects
PATCH_FORMAT instructions (SEARCH/REPLACE etc.) automatically and the model
is expected to emit a patch instead of the full file; the proxy usually
reconstructs the full file transparently (Option A) before this module ever
sees the response. When it can't (its own in-memory artifact — derived from
conversation text — has drifted from the real file on disk, so it can't
resolve which artifact the patch targets), the raw patch text is passed
through untouched instead; _try_apply_raw_patch detects that case and
applies the patch itself directly against the real file, using the same
pure functions from patch_engine.py the proxy uses.

Non-file actions (running a command, deleting a file, reading a file not
yet in context, listing a directory) use a small custom text-tag protocol
that similarly avoids `tools`/`tool_calls`:

    <alfa1:run_command>npm test</alfa1:run_command>
    <alfa1:delete_file>old/module.py</alfa1:delete_file>
    <alfa1:read_file>src/util.py</alfa1:read_file>
    <alfa1:list_files>src</alfa1:list_files>
"""
from __future__ import annotations

import json
import re
from typing import Awaitable, Callable

import httpx

import alfa1_tools
from alfa1_config import get_self_base_url
from alfa1_tools import Alfa1Error
from patch_engine import _ANY_PATCH_MARKER, apply_patch_ops, is_patch_response, parse_patch_response
from transformers import _FILENAME_COMMENT, _FILENAME_HEADER, _extract_code_blocks

_MAX_ITERATIONS = 8

ALFA1_SYSTEM_PROMPT = """\
You are Alfa1, an autonomous coding agent working directly in the user's \
project folder through the ShapeShifter proxy. You act by producing plain \
text — you do NOT have a function/tool-calling API available. Do not emit \
ANY function-calling syntax of any kind (no JSON tool_calls, no <|...|> \
tokens, no "invoke"/"parameter" XML-ish blocks, nothing your training may \
have taught you as a generic tool-call format). The ONLY way to act is the \
plain-text conventions below — nothing else will be understood or executed.

To create or overwrite a file: on its own line immediately BEFORE a fenced \
code block, write the file's path relative to the project root wrapped in \
backticks, then the fenced code block with the file's exact content. Example:

`src/app.py`
```python
print("hello")
```

If you are shown PATCH_FORMAT instructions for a file you've already \
written, use that patch format instead of repeating the whole file.

To run a shell command in the project root:
<alfa1:run_command>the command</alfa1:run_command>

To read a file you have not seen the contents of yet:
<alfa1:read_file>path/relative/to/root</alfa1:read_file>

To delete a file or directory:
<alfa1:delete_file>path/relative/to/root</alfa1:delete_file>

To list a directory (use "." for the project root):
<alfa1:list_files>path/relative/to/root</alfa1:list_files>

To search for text across all files in the project (plain words or a regex):
<alfa1:search_files>text to find</alfa1:search_files>

Only use the actions you actually need this turn. When your work is done, \
reply normally with no further actions or code blocks.\
"""

_ACTION_TAG_RE = re.compile(
    r'<alfa1:(run_command|delete_file|read_file|list_files|search_files)>([\s\S]*?)</alfa1:\1>',
)

# Some models fall back to a self-invented pseudo tool-calling syntax (seen in
# the wild: OpenRouter-routed models emitting things like
# `<|DSML|invoke name="read_file"> <|DSML|parameter name="path">x</...>`)
# even when no `tools` field is declared on the request and the system prompt
# explicitly forbids it. Left unrecognized, this shows up as confusing raw
# garbage in the chat log instead of either doing something or failing
# clearly. Detect the attempt (without trying to fully parse every possible
# dialect) so the turn can be corrected instead of silently treated as a
# normal final answer.
_UNRECOGNIZED_TOOL_ATTEMPT_RE = re.compile(
    r'<\|[\w-]+\|>|\binvoke\s+name\s*=|\btool_calls?\b\s*[:>]|"tool_calls"\s*:',
    re.IGNORECASE,
)

# Best-effort extraction of "what was the model trying to do" from the
# various pseudo tool-call dialects seen in practice (e.g. OpenRouter/DSML-
# style `<|DSML|invoke name="read_file"> <|DSML|parameter name="path">x.py`)
# — used only to make the "unsupported tool-call format" warning legible
# instead of a bare "something went wrong", not to actually execute anything.
_LOOSE_TOOL_NAME_RE = re.compile(r'(?:invoke\s+name|"name")\s*[=:]\s*"?([A-Za-z_][\w]*)"?', re.IGNORECASE)
_LOOSE_TOOL_ARG_RE = re.compile(r'>\s*([^<>{}\n]*\S[^<>{}\n]{0,200})\s*<', re.IGNORECASE)


def _describe_tool_attempt(attempted_tool_calls, content: str) -> str:
    """Human-readable summary of a rejected tool-call attempt, shown in the
    UI's warning banner so the user can see what the model was going for
    instead of just "an unsupported format was used"."""
    if attempted_tool_calls:
        calls = attempted_tool_calls if isinstance(attempted_tool_calls, list) else [attempted_tool_calls]
        parts = []
        for call in calls:
            fn = call.get("function", call) if isinstance(call, dict) else {}
            name = fn.get("name", "?") if isinstance(fn, dict) else "?"
            args = fn.get("arguments", "") if isinstance(fn, dict) else ""
            parts.append(f"{name}({args})" if args else f"{name}()")
        return "Tried to call: " + ", ".join(parts)

    name_m = _LOOSE_TOOL_NAME_RE.search(content)
    if name_m:
        arg_m = _LOOSE_TOOL_ARG_RE.search(content, name_m.end())
        arg = f"({arg_m.group(1).strip()})" if arg_m else "(...)"
        return f"Tried to call: {name_m.group(1)}{arg}"

    snippet = content.strip().replace("\n", " ")
    if len(snippet) > 160:
        snippet = snippet[:160] + "…"
    return f"Raw reply: {snippet}" if snippet else "(empty reply)"


def _strip_fence(block: str) -> str:
    lines = block.splitlines()
    if len(lines) < 2:
        return ""
    inner = lines[1:-1]
    return "\n".join(inner) + ("\n" if inner else "")


# patch_engine.reconstruct_full_file_response (patch_engine.py:376-382) always
# emits this exact status line followed by ONE fence containing the complete
# reconstructed file, as the LAST thing in the message with nothing after it.
# Matched separately, greedily, anchored to end-of-string — NOT via the
# generic non-greedy _extract_code_blocks scan below — because the model's
# own prose before this marker can itself contain a stray, unclosed ``` (seen
# in practice: a model starting its own `path`/```lang file-write convention
# and then abandoning it mid-way to emit SEARCH/REPLACE patch markers
# instead). An odd total number of ``` markers makes non-greedy pairing latch
# onto the wrong two fences and capture only a fragment of the status line as
# "the file" — silently truncating the real content to a couple of dozen
# bytes. Anchoring to the marker and to the end of the string sidesteps that
# ambiguity entirely, since ShapeShifter itself controls this exact shape.
_SHAPESHIFTER_RECONSTRUCTED_RE = re.compile(
    r'\[ShapeShifter:[^\]\n]*\]\s*—\s*complete updated file:\s*\n+```(\w*)\n([\s\S]*)\n```\s*$',
)


def _extract_file_writes(content: str) -> list[tuple[str, str]]:
    """Find (path, content) pairs from fenced code blocks.

    Two filename conventions are recognized, matching transformers.py's own
    _artifact_key heuristics exactly (in the same priority order) so a file
    Alfa1 writes is tracked as the same artifact ShapeShifter itself sees:

    1. A `path.ext`-style header line immediately before the fence — the
       convention ALFA1_SYSTEM_PROMPT asks the model to use directly.
    2. A `# path.ext` (or `//`, `/* */`, `<!--`) comment as the first line
       INSIDE the fence — this is how patch_engine.reconstruct_full_file_response
       labels a patched file (`# {artifact_key}`) when it hands back the
       fully-reconstructed content after applying a SEARCH/REPLACE patch, so
       this case must be handled too or every patch-mode edit would be
       silently dropped. That marker line is metadata, not real file
       content, so it is stripped before writing.
    """
    m = _SHAPESHIFTER_RECONSTRUCTED_RE.search(content)
    if m:
        body_text = m.group(2)
        if is_patch_response(body_text):
            # The "reconstruction" is not trustworthy: it still contains
            # unresolved patch markers. Observed in practice — a model
            # imitated the "[ShapeShifter: N patches applied]" status line
            # itself (likely echoing a pattern it had seen earlier in its
            # own conversation history) while ALSO tacking on a second,
            # genuinely unresolved patch attempt after it. Because this
            # regex matches greedily to the end of the string (see the
            # comment above), that trailing raw patch got swallowed into
            # "body" and would otherwise be written to disk verbatim,
            # <<<<<<< SEARCH markers and all. Defer to run_agent_turn's
            # _try_apply_raw_patch instead, which parses whatever
            # well-formed patch ops exist anywhere in `content` and applies
            # them against the real current file — safe regardless of
            # whatever else the model wrote around them.
            return []
        body_lines = body_text.splitlines()
        path = None
        if body_lines:
            m2 = _FILENAME_COMMENT.match(body_lines[0])
            if m2:
                path = m2.group(1)
                body_lines = body_lines[1:]
        if path is None:
            hm = _FILENAME_HEADER.search(content[:m.start()][-200:])
            if hm:
                path = hm.group(1)
        if path:
            body = "\n".join(body_lines)
            return [(path, body + ("\n" if body else ""))]
        return []

    writes: list[tuple[str, str]] = []
    for block in _extract_code_blocks(content):
        idx = content.find(block)
        preceding = content[:idx] if idx >= 0 else ""
        inner = _strip_fence(block)

        # Defense in depth: a fenced block whose body is itself an unresolved
        # patch (raw <<<<<<< SEARCH markers etc., not yet applied against any
        # real content) is never literal file content — writing it verbatim
        # would corrupt the file with patch syntax instead of code. The
        # actual application of such a patch is run_agent_turn's job (see
        # _try_apply_raw_patch), which runs before this function is called;
        # this guard just ensures the property holds regardless of caller.
        if is_patch_response(inner):
            continue

        m = _FILENAME_HEADER.search(preceding[-200:])
        if m:
            writes.append((m.group(1), inner))
            continue

        inner_lines = inner.splitlines()
        if inner_lines:
            m2 = _FILENAME_COMMENT.match(inner_lines[0])
            if m2:
                remainder = "\n".join(inner_lines[1:])
                if remainder:
                    remainder += "\n"
                writes.append((m2.group(1), remainder))
    return writes


def _resolve_patch_target_path(content: str, marker_pos: int) -> str | None:
    """Best-effort filename for a raw (unresolved-by-the-proxy) patch
    response: the same header/comment conventions used for file writes,
    falling back to "the only file in the workspace" when the workspace has
    exactly one.

    Regression note: the patch marker sits INSIDE the fence (header ->
    ```lang -> <<<<<<< SEARCH), one or more lines after the filename header.
    Searching for the header immediately before marker_pos therefore always
    misses it — the fence-open line sits between them, breaking
    _FILENAME_HEADER's "must be immediately before end of string" anchor.
    Observed in practice causing every apply_patch attempt in a session to
    fail with "didn't clearly specify which file it targets", repeatedly,
    even though the model correctly wrote `path.ext` before every attempt.
    Look for the header right before the FENCE (like _extract_file_writes
    does for normal writes), not right before the marker.
    """
    fence_start = content.rfind("```", 0, marker_pos)
    if fence_start != -1:
        m = _FILENAME_HEADER.search(content[:fence_start][-200:])
        if m:
            return m.group(1)
        m = _FILENAME_COMMENT.search(content[fence_start:marker_pos][:200])
        if m:
            return m.group(1)

    preceding = content[:marker_pos]
    m = _FILENAME_HEADER.search(preceding[-200:])
    if m:
        return m.group(1)
    m = _FILENAME_COMMENT.search(preceding[-200:])
    if m:
        return m.group(1)
    try:
        entries = alfa1_tools.list_tree(".")
    except Alfa1Error:
        return None
    files = [e["path"] for e in entries if e["type"] == "file"]
    return files[0] if len(files) == 1 else None


def _try_apply_raw_patch(content: str) -> tuple[str | None, str, str | None] | None:
    """Detect a raw SEARCH/REPLACE (or REPLACE_FUNCTION/INSERT_AFTER/etc.)
    patch that the proxy did NOT already resolve and reconstruct into a full
    file (see _SHAPESHIFTER_RECONSTRUCTED_RE) — e.g. because ShapeShifter's
    own in-memory artifact (derived from conversation text) had drifted out
    of sync with the real file on disk, so `resolve_target_artifact` failed
    and the raw patch text was passed through untouched. Left alone, that
    raw text (containing literal `<<<<<<< SEARCH` markers) would otherwise
    be picked up by the generic file-write scan below and written to disk
    VERBATIM as if it were the file's real content — a real bug observed in
    practice, corrupting the file with patch syntax instead of code.

    Returns None if there's no unresolved patch here (nothing to do — either
    no patch markers at all, or the proxy already reconstructed a clean,
    trustworthy full file — see _extract_file_writes' matching check for
    why "trustworthy" isn't automatic just because the marker is present).
    Otherwise returns (path, outcome_message, new_full_content); path is
    None if no target file could be determined, new_full_content is None
    unless the patch actually succeeded.

    The caller MUST overwrite the stored assistant message's content with
    new_full_content (via _format_file_for_context) when it's not None —
    otherwise the conversation history still shows the model's raw,
    unresolved <<<<<<< SEARCH text as "what calculator.py currently looks
    like". transformers.py's own artifact tracking (_extract_latest_artifacts)
    scans conversation TEXT for the latest code block per file, with no
    knowledge that this module silently patched the real file out-of-band —
    so every later turn would keep being shown stale patch syntax as
    "current_artifacts" instead of the real file, a real bug observed in
    practice (the model got visibly confused about which version — 799
    bytes or 939 — was actually current, going back and forth across
    several turns instead of making progress).
    """
    reconstructed = _SHAPESHIFTER_RECONSTRUCTED_RE.search(content)
    if reconstructed and not is_patch_response(reconstructed.group(2)):
        return None
    m = _ANY_PATCH_MARKER.search(content)
    if not m:
        return None

    path = _resolve_patch_target_path(content, m.start())
    if not path:
        return None, (
            "Your patch didn't clearly specify which file it targets, so it "
            "could not be applied automatically. Mention the file's path "
            "explicitly (e.g. a `path.ext` header right before the patch) "
            "and resend it."
        ), None
    try:
        current = alfa1_tools.read_file(path)
    except Alfa1Error as exc:
        return path, f"Could not read {path} to apply the patch: {exc}", None
    if current.get("binary"):
        return path, f"{path} is a binary file — cannot apply a text patch to it.", None

    ops = parse_patch_response(content)
    new_text, ok, failed = apply_patch_ops(current["content"], ops)
    if ok == 0:
        return path, (
            f"None of the {failed} patch operation(s) matched the current "
            f"content of {path} exactly — your SEARCH text differs from it "
            f"somewhere (even a single missing or extra line is enough). "
            f"Retry with a new SEARCH/REPLACE patch copied verbatim from "
            f"the exact content below — do NOT rewrite the whole file, "
            f"the change is still small enough for a patch:\n\n"
            f"{_format_file_for_context(path, current['content'])}"
        ), None

    problem = _verify_content(path, new_text)
    if problem:
        return path, (
            f"Patch application produced invalid content and was rejected: "
            f"{problem}. The file was NOT modified. Current content:\n\n"
            f"{_format_file_for_context(path, current['content'])}"
        ), None

    alfa1_tools.write_file(path, new_text)
    status = f"{ok} patch{'es' if ok != 1 else ''} applied to {path}"
    if failed:
        status += f", {failed} failed (kept previous content for those)"
    return path, status, new_text


def _format_file_for_context(path: str, text: str) -> str:
    """Wrap file content the same way a write does, so a file the agent only
    READ still gets picked up by transformers.py's artifact tracking as a
    prior version — required for patch-mode to activate on later edits."""
    ext = path.rsplit(".", 1)[-1] if "." in path else ""
    return f"`{path}`\n```{ext}\n{text}\n```"


def _verify_content(path: str, text: str) -> str | None:
    """Deterministic, free, instant sanity check run BEFORE content is
    written to disk — a cheap "verifier" pass that doesn't need a second LLM
    call. Every real file-corruption bug hit in practice this session (raw
    <<<<<<< SEARCH markers ending up as "the file", a truncated/malformed
    patch producing broken Python) is exactly the class of problem this
    catches. Returns None if the content looks fine, or a description of
    what's wrong — the caller must refuse to write when this is not None,
    reporting the problem back to the model instead of corrupting the file.

    Deliberately NOT a second full LLM review pass ("does this code make
    logical sense") — that's a real possible future enhancement, but this
    layer is about catching mechanical corruption fast and for free, which
    covers everything actually observed going wrong so far.
    """
    if is_patch_response(text):
        return (
            "the content still contains unresolved patch markers "
            "(<<<<<<< SEARCH / >>>>>>> REPLACE etc.) — that looks like a "
            "patch, not the file's actual content"
        )
    if path.endswith(".py"):
        try:
            compile(text, path, "exec")
        except SyntaxError as exc:
            return f"the resulting Python has a syntax error: {exc}"
    if path.endswith(".json"):
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return f"the resulting JSON is invalid: {exc}"
    return None


# Generous default vs. the proxy's own DEFAULT_MAX_OUTPUT_TOKENS (1200) —
# observed in practice truncating mid-response for verbose reasoning models,
# cutting an action tag off before its closing tag (e.g. "<alfa1:read_file>calc"
# with no ".py</alfa1:read_file>"). A truncated tag matches neither
# _ACTION_TAG_RE nor a real file write, so without a larger budget AND the
# finish_reason=="length" handling below, the turn would just silently end
# on garbled text instead of retrying.
_DEFAULT_MAX_TOKENS = 4096


async def _call_self(
    conversation: list[dict], model: str | None,
    on_delta: Callable[[str, str], Awaitable[None]] | None = None,
) -> dict:
    """Streams the completion from this same process's own
    /v1/chat/completions endpoint, invoking on_delta(kind, text) — kind is
    "content" or "reasoning" — as each chunk arrives, so the UI can show
    tokens live instead of a generic "thinking" indicator. Returns the same
    {"choices": [{"message": ..., "finish_reason": ...}]} shape a
    non-streaming call would, by accumulating chunks — so none of the
    post-processing below (patch detection, action parsing, tool_calls
    stripping) needs to know streaming happened at all.

    Note this means the proxy's own patch-reconstruction (Option A) never
    applies here — wrapper_server only reconstructs full-file content for
    non-streaming requests, forwarding raw patch text as-is for streaming
    ones (see wrapper_server._relay_stream's docstring). That's fine: any
    raw, unresolved patch text is handled by _try_apply_raw_patch below
    exactly as it already needs to be for the case where the proxy's own
    reconstruction fails, so streaming just means that path is taken more
    often rather than needing new handling.
    """
    payload = {
        "model": model, "messages": conversation, "stream": True,
        "max_tokens": _DEFAULT_MAX_TOKENS,
    }
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_call_frags: dict[int, dict] = {}
    finish_reason = None

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", f"{get_self_base_url()}/v1/chat/completions", json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta", {})
                if delta.get("content"):
                    content_parts.append(delta["content"])
                    if on_delta is not None:
                        await on_delta("content", delta["content"])
                if delta.get("reasoning"):
                    reasoning_parts.append(delta["reasoning"])
                    if on_delta is not None:
                        await on_delta("reasoning", delta["reasoning"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_call_frags.setdefault(
                        idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]

    message: dict = {"role": "assistant", "content": "".join(content_parts)}
    if reasoning_parts:
        message["reasoning"] = "".join(reasoning_parts)
    if tool_call_frags:
        message["tool_calls"] = [tool_call_frags[i] for i in sorted(tool_call_frags)]
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


async def run_agent_turn(
    conversation: list[dict],
    model: str | None = None,
    max_iterations: int = _MAX_ITERATIONS,
    on_event: Callable[[dict], Awaitable[None]] | None = None,
) -> list[dict]:
    """Runs one user turn to completion. Mutates and returns `conversation`."""

    async def emit(evt: dict) -> None:
        if on_event is not None:
            await on_event(evt)

    async def on_delta(kind: str, text: str) -> None:
        await emit({"type": f"{kind}_delta", "text": text})

    if not conversation or conversation[0].get("role") != "system":
        conversation.insert(0, {"role": "system", "content": ALFA1_SYSTEM_PROMPT})

    for _ in range(max_iterations):
        await emit({"type": "thinking"})
        try:
            resp = await _call_self(conversation, model, on_delta=on_delta)
        except httpx.HTTPError as exc:
            await emit({"type": "error", "message": str(exc)})
            return conversation

        finish_reason = resp["choices"][0].get("finish_reason")
        message = resp["choices"][0]["message"]
        # Some models emit a native tool_calls/function_call structure even
        # though this request never declared any `tools` — observed in
        # practice with deepseek-v4-flash via OpenRouter attempting a
        # `shapeshifter_read_file` call unprompted. Left in place, that field
        # would sit in `conversation` forever and make wrapper_server's
        # _is_agentic() (wrapper_server.py:339-359) treat EVERY subsequent
        # request in this session as agentic passthrough — silently
        # disabling both input compression and output patch savings for the
        # rest of the conversation, with no visible error. Strip it before
        # storing, and treat its presence the same as the unrecognized
        # plain-text tool-call syntax below (it's the same underlying
        # problem, just in structured form instead of leaked text).
        attempted_tool_calls = message.get("tool_calls") or message.get("function_call")
        had_native_tool_call = bool(attempted_tool_calls)
        message.pop("tool_calls", None)
        message.pop("function_call", None)
        conversation.append(message)
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or ""
        if reasoning:
            await emit({"type": "reasoning", "content": reasoning})

        patch_result = _try_apply_raw_patch(content)
        if patch_result is not None:
            path, outcome, new_full_content = patch_result
            await emit({"type": "tool_call", "name": "apply_patch", "arguments": {"path": path}})
            await emit({"type": "tool_result", "name": "apply_patch", "result": outcome})
            if new_full_content is not None:
                # Keep stored history consistent with the real file: replace
                # the model's raw <<<<<<< SEARCH text with what the file
                # actually looks like now, or ShapeShifter's own artifact
                # tracking would keep showing stale patch syntax as "current"
                # on every later turn (see _try_apply_raw_patch's docstring).
                message["content"] = _format_file_for_context(path, new_full_content)
            conversation.append({
                "role": "user",
                "content": "[alfa1] Action results from your last turn:\n\n" + outcome,
            })
            continue

        file_writes = _extract_file_writes(content)
        actions = _ACTION_TAG_RE.findall(content)

        if not file_writes and not actions:
            if had_native_tool_call or _UNRECOGNIZED_TOOL_ATTEMPT_RE.search(content):
                description = _describe_tool_attempt(attempted_tool_calls, content)
                await emit({"type": "tool_attempt_unrecognized", "content": content, "description": description})
                conversation.append({
                    "role": "user",
                    "content": (
                        "[alfa1] Your last reply used an unsupported tool-calling "
                        "syntax that could not be executed. You do not have a "
                        "function-calling API. Use ONLY the plain-text conventions "
                        "from your instructions (backtick-path + code fence for "
                        "files, <alfa1:action>...</alfa1:action> tags for other "
                        "actions) — nothing else is understood."
                    ),
                })
                continue
            if finish_reason == "length":
                # Cut off mid-generation before completing a file write or an
                # <alfa1:action> tag — e.g. "<alfa1:read_file>calc" with no
                # closing ".py</alfa1:read_file>". That's neither a valid
                # action nor recognizable garbage, so without this check it
                # would silently become the "final answer" the user sees.
                await emit({"type": "truncated", "content": content})
                conversation.append({
                    "role": "user",
                    "content": (
                        "[alfa1] Your last reply was cut off before it finished "
                        "(hit the output length limit) — it did not complete an "
                        "action. Retry with a shorter, more direct response: skip "
                        "restating your reasoning and go straight to the action "
                        "(file write or <alfa1:action> tag)."
                    ),
                })
                continue
            await emit({"type": "assistant", "content": content})
            return conversation

        results: list[str] = []

        for path, file_content in file_writes:
            await emit({"type": "tool_call", "name": "write_file", "arguments": {"path": path}})
            problem = _verify_content(path, file_content)
            if problem:
                results.append(
                    f"Refused to write {path}: {problem}. The file was NOT "
                    f"modified — resend the correct content."
                )
            else:
                try:
                    r = alfa1_tools.write_file(path, file_content)
                    results.append(f"Wrote {path} ({r['bytes_written']} bytes)")
                except Alfa1Error as exc:
                    results.append(f"Error writing {path}: {exc}")
            await emit({"type": "tool_result", "name": "write_file", "result": results[-1]})

        for name, arg in actions:
            arg = arg.strip()
            await emit({"type": "tool_call", "name": name, "arguments": {"arg": arg}})
            try:
                if name == "run_command":
                    r = await alfa1_tools.run_command(arg)
                    outcome = (
                        f"$ {arg}\nexit={r['exit_code']} timed_out={r['timed_out']}\n"
                        f"stdout:\n{r['stdout']}\nstderr:\n{r['stderr']}"
                    )
                elif name == "delete_file":
                    alfa1_tools.delete_file(arg)
                    outcome = f"Deleted {arg}"
                elif name == "read_file":
                    r = alfa1_tools.read_file(arg)
                    if r.get("binary"):
                        outcome = f"{arg} is a binary file — cannot show its content."
                    else:
                        outcome = f"Content of {_format_file_for_context(arg, r['content'])}"
                elif name == "list_files":
                    entries = alfa1_tools.list_tree(arg or ".")
                    listing = "\n".join(f"{e['type']}: {e['path']}" for e in entries)
                    outcome = f"Listing of '{arg or '.'}':\n{listing}"
                elif name == "search_files":
                    matches = alfa1_tools.search_files(arg)
                    if not matches:
                        outcome = f"No matches for '{arg}'."
                    else:
                        lines = [f"{m['path']}:{m['line']}: {m['text']}" for m in matches]
                        outcome = f"Found {len(matches)} match(es) for '{arg}':\n" + "\n".join(lines)
                else:
                    outcome = f"Unknown action: {name}"
            except Alfa1Error as exc:
                outcome = f"Error in {name}({arg}): {exc}"
            results.append(outcome)
            await emit({"type": "tool_result", "name": name, "result": outcome})

        conversation.append({
            "role": "user",
            "content": "[alfa1] Action results from your last turn:\n\n" + "\n\n".join(results),
        })

    last_assistant = next(
        (m.get("content", "") for m in reversed(conversation) if m.get("role") == "assistant"), "",
    )
    await emit({"type": "assistant", "content": last_assistant or "(stopped: reached max iterations)"})
    return conversation
