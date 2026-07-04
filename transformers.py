# Copyright (c) 2026 Gaetano Marcello Incarbone. MIT License — see LICENSE file.
"""Context transformers — one function per mode.

Each transformer receives the raw concatenated context string and returns a
compressed/restructured string. No LLM calls here: all transforms are
deterministic heuristics for MVP 0.1/0.2.
"""
from __future__ import annotations

import re
import json
import yaml  # type: ignore[import]
from typing import Callable

# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

_ERROR_PATTERNS = re.compile(
    r"(ERROR|Exception|Traceback|FATAL|WARN|Caused by|at\s+\w+\.\w+|"
    r"SyntaxError|TypeError|ValueError|NullPointer|StackOverflow)",
    re.IGNORECASE,
)
_FILE_PATTERNS = re.compile(r'\b[\w/\\.-]+\.\w{2,5}\b')
_CODE_BLOCK = re.compile(r'```[\s\S]*?```', re.DOTALL)


def _extract_error_lines(text: str, max_lines: int = 20) -> list[str]:
    return [ln for ln in text.splitlines() if _ERROR_PATTERNS.search(ln)][:max_lines]


def _extract_filenames(text: str) -> list[str]:
    found = _FILE_PATTERNS.findall(text)
    seen: set[str] = set()
    out = []
    for f in found:
        if f not in seen and not f.startswith("http"):
            seen.add(f)
            out.append(f)
    return out[:10]


def _extract_code_blocks(text: str) -> list[str]:
    return _CODE_BLOCK.findall(text)


def _head_tail(text: str, head: int = 8, tail: int = 8) -> str:
    lines = text.splitlines()
    if len(lines) <= head + tail:
        return text
    return "\n".join(lines[:head] + ["..."] + lines[-tail:])


def _key_sentences(text: str, max_sentences: int = 12) -> list[str]:
    """Return sentences most likely to be informative (short, contain nouns/verbs)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    scored = sorted(sentences, key=lambda s: (
        -int(bool(_ERROR_PATTERNS.search(s))),   # errors first
        len(s)                                    # shorter preferred
    ))
    return scored[:max_sentences]


def _infer_task(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ("debug", "error", "exception", "fix")):
        return "debug_or_fix"
    if any(k in lower for k in ("analysis", "explain")):
        return "analysis"
    if any(k in lower for k in ("compare", "comparison", "difference")):
        return "comparison"
    if any(k in lower for k in ("generate", "create", "write",
                                 "add", "build", "implement", "extend")):
        return "generation"
    if any(k in lower for k in ("refactor", "optimize", "improve")):
        return "refactor"
    return "general"


_CODE_SIGNALS = re.compile(
    r'(?:'
    r'```\w*\s*\n'
    r'|def\s+\w+[\s(]|function\s+\w+\s*\(|fn\s+\w+\s*\('
    r'|fun\s+\w+\s*\(|func\s+\w+\s*\(|sub\s+\w+\s*\(|proc\s+\w+\s*\('
    r'|class\s+\w+|struct\s+\w+|interface\s+\w+|enum\s+\w+'
    r'|impl\s+\w+|trait\s+\w+'
    r'|import\s+\w+|from\s+\w+\s+import|require\s*\(|include\s+[<"]'
    r'|use\s+\w+::|using\s+\w+|extern\s+crate'
    r'|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*='
    r'|#include|#define|package\s+\w+|namespace\s+\w+'
    r'|<!DOCTYPE|<html\b|<\?php|<\?xml'
    r')',
    re.IGNORECASE,
)


def _is_coding_session(context: str) -> bool:
    """True only when a previous ASSISTANT turn already contains generated code.

    Checks [ASSISTANT] blocks specifically — not the whole context — to avoid
    false positives when workspace files or user-pasted snippets contain code
    but the conversation itself is a fresh first-turn generation request.
    """
    if not re.search(r'\[ASSISTANT\]', context):
        return False  # no assistant turn yet → not a multi-turn coding session

    assistant_blocks = re.findall(
        r'\[ASSISTANT\]\n([\s\S]*?)(?=\n\n\[(?:USER|ASSISTANT)\]|$)',
        context,
    )
    return any(_CODE_SIGNALS.search(block) for block in assistant_blocks)


def _extract_user_requirements(context: str) -> list[str]:
    """Return [USER] blocks verbatim (no truncation, no stripping).

    User messages may contain pasted code, file excerpts, or examples that are
    essential to answer correctly. Only [ASSISTANT] blocks (generated code) are
    discarded — that is where the token savings come from.
    """
    blocks = re.findall(
        r'\[USER\]\n([\s\S]*?)(?=\n\n\[(?:USER|ASSISTANT)\]|$)',
        context,
    )
    return [b.strip() for b in blocks if b.strip()]


_FENCE_LANG = re.compile(r'```(\w+)?')
_FILENAME_COMMENT = re.compile(
    r'(?:^|\n)\s*(?:#|//|/\*|<!--)\s*(?:file(?:name)?:?\s*)?'
    r'([\w\-./\\]+\.\w{1,10})\b',
    re.IGNORECASE,
)
_FILENAME_HEADER = re.compile(
    r'(?:\*\*|##+|`)\s*([\w\-./\\]+\.\w{1,10})\s*(?:\*\*|`)?\s*\n?\s*$',
)


def _artifact_key(code_block: str, preceding_text: str) -> str:
    """Best-effort identifier for the file/artifact a code block represents,
    so later versions of the SAME file can supersede earlier ones instead of
    being treated as unrelated blocks.

    Tries, in order: a filename comment inside the block, a filename mentioned
    in the assistant text right before the block (e.g. "**app.py**" or
    "`app.py`:"), then falls back to the fence language — which correctly
    collapses the common single-file iterative-build session (one Python or
    HTML file revised turn after turn) into one artifact even with no
    filename ever mentioned.
    """
    m = _FILENAME_COMMENT.search(code_block[:200])
    if m:
        return m.group(1).lower()
    m = _FILENAME_HEADER.search(preceding_text[-200:])
    if m:
        return m.group(1).lower()
    m = _FENCE_LANG.match(code_block)
    lang = (m.group(1) or "code").lower() if m else "code"
    return f"__lang__:{lang}"


def _extract_latest_artifacts(context: str) -> dict[str, str]:
    """Walk [USER] and [ASSISTANT] blocks in chronological order and keep only
    the LAST code block seen per artifact key, regardless of which role
    produced it. Earlier versions of the same file are superseded and
    dropped — but unlike dropping all generated code, the model still gets
    the actual current state of every file it has touched, which is required
    to make a precise edit or fix a bug instead of regenerating from scratch.

    A user pasting the current state of a file they edited by hand (or an
    error dump with the file attached) supersedes an earlier assistant draft
    just as much as a newer assistant turn would — what matters is which
    version is chronologically last, not which role wrote it.
    """
    blocks = re.findall(
        r'\[(?:USER|ASSISTANT)\]\n([\s\S]*?)(?=\n\n\[(?:USER|ASSISTANT)\]|$)',
        context,
    )
    latest: dict[str, str] = {}
    for block in blocks:
        for code in _extract_code_blocks(block):
            idx = block.find(code)
            preceding = block[:idx] if idx >= 0 else ""
            key = _artifact_key(code, preceding)
            latest[key] = code  # overwritten on re-occurrence — last write wins
    return latest


def _extract_artifact_versions(context: str) -> dict[str, list[str]]:
    """Like `_extract_latest_artifacts`, but keeps the last TWO versions per
    key (previous, current) instead of only the latest — needed to know
    which top-level blocks are unchanged between them (see
    `_collapse_unchanged_blocks`). Returns `[current]` for a key seen only
    once (nothing to diff against yet).
    """
    blocks = re.findall(
        r'\[(?:USER|ASSISTANT)\]\n([\s\S]*?)(?=\n\n\[(?:USER|ASSISTANT)\]|$)',
        context,
    )
    versions: dict[str, list[str]] = {}
    for block in blocks:
        for code in _extract_code_blocks(block):
            idx = block.find(code)
            preceding = block[:idx] if idx >= 0 else ""
            key = _artifact_key(code, preceding)
            history = versions.setdefault(key, [])
            history.append(code)
            if len(history) > 2:
                history.pop(0)
    return versions


_DEF_LINE = re.compile(
    r'^\s*(?:async\s+|export\s+|default\s+|pub\s+)*(?:'
    r'def\s+\w+|function\s+\w+|fn\s+\w+|fun\s+\w+|func\s+\w+|sub\s+\w+|proc\s+\w+'
    r'|class\s+\w+|struct\s+\w+|interface\s+\w+|enum\s+\w+|impl\s+\w+|trait\s+\w+'
    r')',
    re.IGNORECASE,
)
_DECORATOR_LINE = re.compile(r'^\s*@\w')


def _split_definition_blocks(code: str) -> list[tuple[str | None, str]]:
    """Split code into (header, full_block_text) chunks at EVERY function/
    method/type declaration line, regardless of indentation — this is what
    gives per-method granularity inside a class instead of treating the
    whole class as one indivisible block. Reuses the same multi-language
    keyword set as `_CODE_SIGNALS` (Python `def`, JS/TS `function`, Rust
    `fn`, Go `func`, Kotlin/Swift `fun`, plus `class`/`struct`/`interface`/
    `enum`/`impl`/`trait`) rather than a single language's syntax, so this
    isn't Python-only — plus common modifiers that precede a declaration
    (`async`, JS/TS `export`/`export default`, Rust `pub`) so `async def`,
    `export function`, `pub fn`, etc. are still recognized as the
    declaration they modify. Decorators/annotations (`@Something` — Python
    decorators and Java/C#-style annotations share the same syntax) directly
    above a declaration are kept attached to that block, not left dangling
    in the previous one. `header` is `None` for the shared preamble before
    the first declaration.

    This deliberately doesn't do real brace-matching or indentation-depth
    tracking — a block is just "everything from this declaration line up to
    the next one." That's exactly right for flat, sequential declarations
    (the overwhelming common case for generated code) but under-splits a
    declaration nested inside another one (e.g. a helper function defined
    inside a method): the outer declaration's block gets cut short at the
    inner one's start, and the inner block absorbs the outer's remaining
    body. This doesn't create a correctness risk — `_collapse_unchanged_blocks`
    still only collapses on exact text equality of whatever range was
    captured — but the block LABEL in that case describes less than what it
    actually contains. A class's own header line (e.g. `class Foo:`) becomes
    its own near-empty block since the very next line already starts the
    first method's block — harmless, and deliberately never collapsed (see
    `_collapse_unchanged_blocks`'s header-only guard) since there's nothing
    worth collapsing in a single line.

    Returns `[]` if no declaration line is recognized at all — callers must
    treat that as "can't confidently identify blocks here" and keep the
    whole file in full rather than guessing at boundaries in an unrecognized
    language or style (e.g. JS arrow functions, Bash `foo() { }`).
    """
    lines = code.splitlines(keepends=True)
    def_starts = [i for i, ln in enumerate(lines) if _DEF_LINE.match(ln)]
    if not def_starts:
        return []

    starts: list[int] = []
    for i in def_starts:
        s = i
        while s > 0 and _DECORATOR_LINE.match(lines[s - 1]):
            s -= 1
        starts.append(s)
    starts = sorted(set(starts))

    blocks: list[tuple[str | None, str]] = []
    if starts[0] > 0:
        blocks.append((None, "".join(lines[:starts[0]])))
    for i, start in enumerate(starts):
        if i + 1 < len(starts):
            end = starts[i + 1]
            header_idx = next(j for j in range(start, end) if _DEF_LINE.match(lines[j]))
        else:
            # Last declaration: bound its body by indentation instead of
            # running to end-of-file, so trailing top-level code (a
            # `if __name__ == "__main__":` block, module-level calls) isn't
            # silently absorbed into it — see Feature 5 in
            # docs/token_savings_roadmap.md. Measure indentation from the
            # actual declaration line, not `start` — `start` may point at a
            # decorator line backed up to keep it attached to this block.
            header_idx = next(j for j in range(start, len(lines)) if _DEF_LINE.match(lines[j]))
            header_indent = len(lines[header_idx]) - len(lines[header_idx].lstrip(" \t"))
            end = _find_block_end_by_indent(lines, header_idx, header_indent)
        block_text = "".join(lines[start:end])
        header = lines[header_idx].rstrip("\n")
        blocks.append((header, block_text))
        if i + 1 == len(starts) and end < len(lines):
            blocks.append((None, "".join(lines[end:])))
    return blocks


_LONE_CLOSER = re.compile(r'^[)\]}]+[;,]?$')


def _find_block_end_by_indent(lines: list[str], start: int, header_indent: int) -> int:
    """Scan forward from a declaration's header line for the first non-blank
    line whose indentation is <= the header's own — the boundary between
    "this declaration's own body" and whatever top-level code follows it at
    the same or shallower level. Works for both indentation-delimited
    (Python) and brace-delimited code, since well-formatted generated code
    de-indents when a block ends regardless of language.

    A lone closing brace/bracket/paren (`}`, `);`, etc.) at that same
    indentation is the block's OWN closing delimiter in brace languages, not
    a boundary — it's included in the block and scanning continues past it.
    Returns `len(lines)` if no real boundary is found (the declaration runs
    to the end of the file).
    """
    j = start + 1
    while j < len(lines):
        line = lines[j]
        if not line.strip():
            j += 1
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if indent <= header_indent:
            if _LONE_CLOSER.match(line.strip()):
                j += 1
                continue
            return j
        j += 1
    return len(lines)


def _normalize_for_comparison(text: str) -> str:
    """Normalize away formatting noise that's never semantically significant
    in any mainstream language, for the sole purpose of the equality check
    in `_collapse_unchanged_blocks` — trailing whitespace on a line, and
    runs of consecutive blank lines. Leading (indentation) whitespace is
    deliberately never touched: a real reindentation or structural change
    must still block the collapse. The actual stored/displayed content when
    a block does NOT collapse is always the real, untouched text — this
    function is never used to alter what the model sees, only to decide
    whether two versions count as "unchanged."
    """
    normalized: list[str] = []
    prev_blank = False
    for line in text.splitlines():
        stripped = line.rstrip()
        is_blank = not stripped
        if is_blank and prev_blank:
            continue
        normalized.append(stripped)
        prev_blank = is_blank
    return "\n".join(normalized)


def _collapse_unchanged_blocks(prev: str, current: str) -> str:
    """Collapse function/method/type blocks in `current` that are unchanged
    from a same-named block in `prev` down to a signature-only stub; every
    block that changed, is new, or can't be matched by name stays in full.
    "Unchanged" means equal after normalizing away trailing whitespace and
    blank-line-run differences (see `_normalize_for_comparison`) — neither
    is ever semantically significant, so a block reformatted only in those
    ways still collapses. Any other difference, including a real
    reindentation, still blocks the collapse exactly as before.

    Falls back to returning `current` completely unchanged whenever block
    splitting isn't confident: no recognized declaration keyword found (see
    `_DEF_LINE` — covers Python/JS/TS/Go/Rust/Kotlin-ish syntax and
    class/struct/interface/enum/impl/trait, but not e.g. JS arrow functions
    or Bash-style `foo() { }`), or fewer than 2 blocks total (nothing
    meaningful to collapse).
    """
    curr_blocks = _split_definition_blocks(current)
    if len(curr_blocks) < 2:
        return current

    prev_by_header = {h: b for h, b in _split_definition_blocks(prev) if h is not None}

    out = []
    for header, block_text in curr_blocks:
        if header is None:
            out.append(block_text)  # preamble (imports, module docstring) always kept in full
            continue
        # A block that's just its own header line (e.g. a bare "class Foo:"
        # whose methods are split out as separate blocks right after it) has
        # nothing worth collapsing — skip it rather than emit a redundant stub.
        header_only = block_text.count("\n") <= 1
        unchanged = (
            not header_only and header in prev_by_header
            and _normalize_for_comparison(prev_by_header[header]) == _normalize_for_comparison(block_text)
        )
        if unchanged:
            base_indent = len(header) - len(header.lstrip(" "))
            stub_indent = " " * (base_indent + 4)
            out.append(f"{header}\n{stub_indent}...  # unchanged since previous version\n")
        else:
            out.append(block_text)
    return "".join(out)


def _extract_latest_artifacts_collapsed(context: str) -> dict[str, str]:
    """Like `_extract_latest_artifacts`, but for artifacts where a previous
    version exists and both versions are confidently splittable into
    function/method/type-level blocks (see `_split_definition_blocks` for
    which languages/keywords are recognized), unchanged blocks collapse to a
    signature-only stub — only the regions that actually changed (or are
    new) stay in full. Falls back to the untouched latest version whenever
    that confidence isn't there, so this never discards content it can't
    prove is unchanged.
    """
    versions = _extract_artifact_versions(context)
    result: dict[str, str] = {}
    for key, history in versions.items():
        if len(history) == 2:
            result[key] = _collapse_unchanged_blocks(history[0], history[1])
        else:
            result[key] = history[-1]
    return result


_DECL_NAME = re.compile(
    r'(?:async\s+|export\s+|default\s+|pub\s+)*(?:'
    r'def|function|fn|fun|func|sub|proc|class|struct|interface|enum|impl|trait'
    r')\s+(\w+)',
    re.IGNORECASE,
)


def _clean_declaration_name(header: str) -> str:
    """Extract just the function/type name from a raw declaration header
    line (stripping modifiers, parameters, and trailing `{`/`:`) — a clean
    token the model can echo back in a retrieval-tool call, rather than the
    whole raw header string. Returns "" if no name can be extracted (should
    not happen for a header that already matched `_DEF_LINE`, but callers
    should treat an empty result as "don't register a retrieval key for
    this block" rather than guessing).
    """
    m = _DECL_NAME.search(header)
    return m.group(1) if m else ""


def _extract_retrievable_pieces(context: str) -> dict[str, str]:
    """Inventory of everything Feature 2/3 might collapse in this context,
    keyed the way the retrieval tool (Feature 6) expects: the artifact key
    alone for a whole-file stub, or `f"{artifact_key}#{function_name}"` for
    a collapsed function within it. Lets ShapeShifter answer the model's own
    retrieval-tool calls instantly from data already computed for this
    request, instead of re-deriving what got collapsed at call time.

    Always includes the whole-file entry for every tracked artifact — even
    ones that ended up fully expanded this turn — since the caller doesn't
    know in advance whether Feature 2 will decide to collapse it (that
    depends on the current turn's text, resolved later in the pipeline).
    """
    versions = _extract_artifact_versions(context)
    pieces: dict[str, str] = {}
    for key, history in versions.items():
        current = history[-1]
        pieces[key] = current
        if len(history) == 2:
            prev_headers = {h for h, _ in _split_definition_blocks(history[0]) if h is not None}
            for header, block_text in _split_definition_blocks(current):
                if header is None or header not in prev_headers:
                    continue
                name = _clean_declaration_name(header)
                if name:
                    pieces[f"{key}#{name}"] = block_text
    return pieces


_TOPLEVEL_IDENT = re.compile(
    r'^\s*(?:class|def|function|fn|func|struct|interface|enum)\s+(\w+)',
    re.MULTILINE,
)


def _artifact_identifiers(code: str) -> list[str]:
    """Top-level class/function/type names declared in a code block — used to
    catch references like "the User model" that name an identifier the file
    defines without ever spelling out the filename itself."""
    return _TOPLEVEL_IDENT.findall(code)


def _format_artifacts_block(artifacts: dict[str, str], current_text: str = "") -> list[str]:
    """Render retained artifacts, collapsing ones that don't look relevant to
    the CURRENT turn to a one-line stub instead of their full body.

    An artifact stays fully expanded if: its filename OR any class/function
    it declares is mentioned in the current turn's message (catches "the
    User model" as well as "models.py"), it's the only artifact being
    tracked (nothing to gain by collapsing), or it has no resolvable
    filename at all (a fence-language fallback key — there's no name the
    model could use to ask for it again, so collapsing it would make it
    unrecoverable rather than just deferred). When in doubt this heuristic
    is deliberately biased toward NOT collapsing — the retention mechanism
    exists specifically so an edit turn can see the file it's editing,
    and a false "irrelevant" guess is a much worse failure than a missed
    compression opportunity. Collapsed entries always say so explicitly and
    invite the model to ask again, rather than silently going quiet about a
    file it already knows exists.
    """
    if not artifacts:
        return []
    lines = ["", "current_artifacts (latest version of each file touched so far — "
                  "edit these directly for fixes/changes, do not regenerate from scratch; "
                  "collapsed entries are unchanged, ask to see one again if you need its content):"]
    only_one = len(artifacts) == 1
    current_lower = current_text.lower()
    for key, code in artifacts.items():
        named = not key.startswith("__lang__:")
        label = key if named else f"(unnamed {key.split(':', 1)[1]} file)"
        mentioned = key.lower() in current_lower or any(
            ident.lower() in current_lower for ident in _artifact_identifiers(code)
        )
        active = only_one or not named or mentioned
        stub = f"  --- {label}: {code.count(chr(10)) + 1} lines, unchanged since last shown — ask to see it again if needed ---"
        # Never collapse if the stub itself wouldn't actually be smaller —
        # for a short file the fixed cost of the stub text can outweigh what
        # it replaces (same principle as the tool-read dedup size guard).
        if active or len(stub) >= len(code):
            lines += [f"  --- {label} ---", code]
        else:
            lines.append(stub)
    return lines


def _infer_stack(text: str) -> list[str]:
    stacks = {
        "Python": r'\bpython\b|\.py\b|traceback|pip\b',
        "Java": r'\bjava\b|\.java\b|tomcat|maven|gradle|nullpointerexception',
        "JSP": r'\bjsp\b|\.jsp\b',
        "JavaScript": r'\bjavascript\b|\.js\b|node\.?js|npm\b',
        "TypeScript": r'\btypescript\b|\.ts\b',
        "SQL": r'\bsql\b|\.sql\b|select\s+\*|insert into',
        "Docker": r'\bdocker\b|dockerfile\b|compose\.yml',
        "Kubernetes": r'\bkubernetes\b|\bk8s\b|kubectl\b',
        "React": r'\breact\b|\.tsx\b|jsx\b',
    }
    found = []
    for name, pattern in stacks.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(name)
    return found


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------

def transform_raw(context: str, current_text: str = "") -> str:
    return context


def transform_minimal(context: str, current_text: str = "") -> str:
    error_lines = _extract_error_lines(context)
    files = _extract_filenames(context)

    parts = []
    if error_lines:
        # max 3 error lines, capped at 80 chars each
        parts.append("ERR: " + " | ".join(ln.strip()[:80] for ln in error_lines[:3]))
    if files:
        parts.append("FILES: " + ", ".join(files[:4]))

    # Add first meaningful non-error sentence (task description) truncated
    lines = [ln.strip() for ln in context.splitlines() if ln.strip() and not _ERROR_PATTERNS.search(ln)]
    task_line = next((ln for ln in lines if len(ln) > 20), "")
    if task_line:
        parts.append("TASK: " + task_line[:200])

    parts.append("→ Reply directly and concisely.")
    return "\n".join(parts)


def transform_yaml(context: str, current_text: str = "") -> str:
    # For multi-turn coding sessions: requirements list + latest artifact
    # version only, no task/stack inference (task inference also scans
    # workspace context and produces false positives).
    if _is_coding_session(context):
        reqs = _extract_user_requirements(context)
        artifacts = _extract_latest_artifacts_collapsed(context)
        data: dict = {
            "context_mode": "yaml",
            "cumulative_requirements": reqs,
            "constraint": "Return COMPLETE file. All requirements must be present.",
        }
        out = "context_mode: yaml\n\n" + yaml.dump(data, default_flow_style=False, allow_unicode=True).strip()
        # Appended as plain text rather than YAML scalars — code blocks contain
        # quotes/colons/backticks that make forced YAML escaping unreadable.
        out += "\n".join(_format_artifacts_block(artifacts, current_text))
        return out

    task = _infer_task(context)
    stack = _infer_stack(context)
    errors = [ln.strip() for ln in _extract_error_lines(context, 5)]
    files = _extract_filenames(context)
    code_blocks = _extract_code_blocks(context)

    data = {"task": task}
    if stack:
        data["environment"] = {"stack": stack}
    if errors:
        data["errors"] = errors
    if files:
        data["files"] = files[:5]
    if code_blocks:
        snippet = code_blocks[0][:300] + ("..." if len(code_blocks[0]) > 300 else "")
        data["code_excerpt"] = snippet
    data["expected_output"] = {"format": "concise_answer", "style": "direct"}

    lines = ["context_mode: yaml", ""]
    lines.append(yaml.dump(data, default_flow_style=False, allow_unicode=True).strip())
    return "\n".join(lines)


def transform_json(context: str, current_text: str = "") -> str:
    task = _infer_task(context)
    stack = _infer_stack(context)
    errors = [ln.strip() for ln in _extract_error_lines(context, 5)]
    files = _extract_filenames(context)
    code_blocks = _extract_code_blocks(context)

    data: dict = {"context_mode": "json", "task": task}
    if stack:
        data["environment"] = {"stack": stack}
    if errors:
        data["errors"] = errors
    if files:
        data["files"] = files[:5]
    if code_blocks:
        snippet = code_blocks[0][:300] + ("..." if len(code_blocks[0]) > 300 else "")
        data["code_excerpt"] = snippet
    data["expected_output"] = "concise_direct_answer"
    return json.dumps(data, ensure_ascii=False, indent=2)


def transform_table(context: str, current_text: str = "") -> str:
    task = _infer_task(context)
    stack = _infer_stack(context)
    errors = _extract_error_lines(context, 3)
    files = _extract_filenames(context)

    rows: list[tuple[str, str]] = [
        ("context_mode", "table"),
        ("task", task),
        ("stack", ", ".join(stack) if stack else "unknown"),
        ("files", ", ".join(files[:3]) if files else "—"),
        ("errors", " | ".join(e.strip()[:80] for e in errors) if errors else "—"),
    ]
    header = "| Field | Value |\n|---|---|"
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return header + "\n" + body


def transform_hybrid(context: str, current_text: str = "") -> str:
    # For multi-turn coding sessions: requirements only, no stack inference.
    # Stack detection scans the whole context including workspace files sent by
    # the client, which causes false positives (e.g. ShapeShifter's own Python
    # source appears in context when the user asks to create a JSP).
    if _is_coding_session(context):
        reqs = _extract_user_requirements(context)
        artifacts = _extract_latest_artifacts_collapsed(context)
        parts = [
            "context_mode: hybrid",
            "",
            "task: generation",
            "  constraint: return COMPLETE file, all requirements must be present",
            "",
            "cumulative_requirements:",
        ]
        for i, r in enumerate(reqs, 1):
            parts.append(f"  [{i}]: {r}")
        parts += _format_artifacts_block(artifacts, current_text)
        return "\n".join(parts)

    task = _infer_task(context)
    stack = _infer_stack(context)
    errors = [ln.strip() for ln in _extract_error_lines(context, 5)]
    files = _extract_filenames(context)
    code_blocks = _extract_code_blocks(context)

    summary_sentences = _key_sentences(context, max_sentences=6)
    summary = " ".join(summary_sentences)[:400]

    excerpt = ""
    if code_blocks:
        excerpt = code_blocks[0][:500]
    elif errors:
        excerpt = "\n".join(errors[:5])
    else:
        excerpt = _head_tail(context, head=5, tail=5)

    parts = [
        "context_mode: hybrid",
        "",
        "task:",
        f"  intent: {task}",
        f"  requested_output: concise_answer",
    ]
    if stack:
        parts += ["", "environment:", "  stack:"] + [f"    - {s}" for s in stack]
    if files:
        parts += ["", "files:"] + [f"  - {f}" for f in files[:5]]
    if errors:
        parts += ["", "errors:"] + [f"  - {e[:120]}" for e in errors[:4]]
    parts += [
        "",
        "summary: |",
        "  " + summary.replace("\n", "\n  "),
        "",
        "relevant_excerpt: |",
        "  " + excerpt.replace("\n", "\n  "),
        "",
        "instructions:",
        "  - answer directly",
        "  - do not repeat the full context",
        "  - use compact output",
        "  - show only relevant code or patch",
        "  - max 8 bullets",
    ]
    return "\n".join(parts)


def transform_symbolic(context: str, current_text: str = "") -> str:
    task = _infer_task(context)
    stack = _infer_stack(context)
    errors = _extract_error_lines(context, 3)
    files = _extract_filenames(context)

    env_str = "{" + ", ".join(stack) + "}" if stack else "{?}"
    error_str = " ∧ ".join(
        re.sub(r'\s+', ' ', e.strip())[:60] for e in errors[:3]
    ) if errors else "?"
    file_str = " | ".join(files[:3]) if files else "?"

    lines = [
        f"TASK = {task.upper()}",
        f"ENV = {env_str}",
        f"FILES = [{file_str}]",
        f"ERROR = {error_str if error_str != '?' else 'see_context'}",
        "GOAL = DirectAnswer ∧ CompactOutput",
        "ACTION_HINT = AnalyzeError → ProposeFix → ExplainBriefly",
    ]
    return "\n".join(lines)


def transform_incremental(context: str, current_text: str = "") -> str:
    """For multi-turn coding/generation sessions.

    Keeps all USER requirements across turns (the feature list the model must
    honour) plus the latest version of each generated artifact — only
    superseded versions of a file are discarded, not every ASSISTANT response.
    The model can then edit the current file directly instead of having to
    regenerate it blind from requirements alone.
    """
    # Parse [USER] / [ASSISTANT] blocks produced by apply_transform
    blocks = re.findall(
        r'\[(USER|ASSISTANT)\]\n([\s\S]*?)(?=\n\n\[(?:USER|ASSISTANT)\]|$)',
        context,
    )

    user_reqs: list[str] = []
    for role, content in blocks:
        if role == "USER":
            c = content.strip()
            if c:
                user_reqs.append(c)  # verbatim: user may paste code as part of their request

    if not user_reqs:
        # Fallback: treat full context as single requirement
        return f"REQUIREMENT:\n{context[:600]}\n\nReturn complete working code."

    artifacts = _extract_latest_artifacts_collapsed(context)

    lines = [
        "CODING_SESSION: incremental",
        "RULE: Every requirement listed below MUST be present in the final output.",
        "",
        "REQUIREMENTS (in order — all cumulative):",
    ]
    for i, req in enumerate(user_reqs, 1):
        lines.append(f"\n[{i}] {req}")

    lines += _format_artifacts_block(artifacts, current_text)
    lines += [
        "",
        "CONSTRAINT: Return the COMPLETE updated file. No truncation. No placeholders.",
    ]
    return "\n".join(lines)


def transform_matrix(context: str, current_text: str = "") -> str:
    task = _infer_task(context)
    stack = _infer_stack(context)
    errors = _extract_error_lines(context, 4)
    files = _extract_filenames(context)
    code_blocks = _extract_code_blocks(context)

    lines = ["ENTITY_MATRIX", ""]
    lines += [f"[TASK]\n  type: {task}", ""]

    for f in files[:3]:
        lines += [f"[FILE] {f}\n  role: target\n", ""]

    if errors:
        for i, e in enumerate(errors[:3]):
            lines += [f"[ERROR_{i+1}]\n  message: {e.strip()[:100]}\n", ""]

    if stack:
        lines += ["[ENVIRONMENT]", "  stack: " + " | ".join(stack), ""]

    if code_blocks:
        snippet = code_blocks[0][:200].strip()
        lines += [f"[CODE_EXCERPT]\n  content: |\n    {snippet.replace(chr(10), chr(10)+'    ')}", ""]

    lines += [
        "[GOAL]",
        "  output: concise_answer",
        "  constraints: no_repetition | max_8_bullets",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRANSFORMERS: dict[str, Callable[[str, str], str]] = {
    "raw":         transform_raw,
    "minimal":     transform_minimal,
    "yaml":        transform_yaml,
    "json":        transform_json,
    "table":       transform_table,
    "hybrid":      transform_hybrid,
    "symbolic":    transform_symbolic,
    "matrix":      transform_matrix,
    "incremental": transform_incremental,
}

VALID_MODES = set(TRANSFORMERS.keys())


def apply_transform(mode: str, messages: list[dict], current_text: str = "") -> tuple[str, str]:
    """Return (original_context, transformed_context).

    `current_text` is the CURRENT turn's user message (not part of
    `messages`/history) — coding-session modes use it to decide which
    retained artifacts are relevant to this turn (see `_format_artifacts_block`).
    Modes that don't need it simply ignore the argument.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown context mode: {mode!r}. Valid: {sorted(VALID_MODES)}")

    raw = "\n\n".join(
        f"[{m['role'].upper()}]\n{m.get('content', '')}"
        for m in messages
        if isinstance(m.get("content"), str)
    )
    transformed = TRANSFORMERS[mode](raw, current_text)
    return raw, transformed
