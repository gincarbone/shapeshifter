# Copyright (c) 2026 Gaetano Marcello Incarbone. MIT License — see LICENSE file.
"""Output patch engine — parse SEARCH/REPLACE and named-entity patches returned by
the model, apply them to an in-memory artifact, and measure output token savings.

Patch format the model is instructed to produce (see PATCH_FORMAT_INSTRUCTIONS):

  Standard search/replace (works at any granularity — preferred for short edits):
      <<<<<<< SEARCH
      [exact lines to replace, verbatim from the file content]
      =======
      [replacement lines]
      >>>>>>> REPLACE

  Named function/method replacement (whole-function rewrite):
      REPLACE_FUNCTION: function_name
      ```lang
      [complete new function body including the def/signature line]
      ```

  Named class replacement (whole-class rewrite):
      REPLACE_CLASS: ClassName
      ```lang
      [complete new class body including the class line]
      ```

  Insert new code after a named entity:
      INSERT_AFTER: existing_name
      ```lang
      [new code block to insert after that entity]
      ```

  Single-line replacement:
      EDIT_LINE: N
      new content for that line

Multiple patch blocks may appear in a single response and are applied in order.
For a brand-new file with no prior version in context, the model returns a complete
code block as usual (no patch needed — the engine detects this and returns 0 savings).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Regex patterns for each patch format
# ---------------------------------------------------------------------------

_SEARCH_REPLACE_RE = re.compile(
    r'<{7}\s*SEARCH\s*\n([\s\S]*?)\n={7}\s*\n([\s\S]*?)\n>{7}\s*REPLACE',
    re.MULTILINE,
)

_REPLACE_FUNC_RE = re.compile(
    r'REPLACE_(?:FUNCTION|METHOD):\s*(\w+)\s*\n```\w*\n([\s\S]*?)```',
    re.MULTILINE,
)

_REPLACE_CLASS_RE = re.compile(
    r'REPLACE_CLASS:\s*(\w+)\s*\n```\w*\n([\s\S]*?)```',
    re.MULTILINE,
)

_INSERT_AFTER_RE = re.compile(
    r'INSERT_AFTER:\s*(\w+)\s*\n```\w*\n([\s\S]*?)```',
    re.MULTILINE,
)

_EDIT_LINE_RE = re.compile(
    r'EDIT_LINE:\s*(\d+)\s*\n(.+)',
    re.MULTILINE,
)

_ANY_PATCH_MARKER = re.compile(
    r'<{7}\s*SEARCH|REPLACE_(?:FUNCTION|METHOD|CLASS):\s*\w|INSERT_AFTER:\s*\w|EDIT_LINE:\s*\d+',
)

# Opening code fence: ```lang or just ```
_OPEN_FENCE = re.compile(r'^```(\w*)\s*$')

# ---------------------------------------------------------------------------
# Patch instructions injected into the prompt (used by transformers.py)
# ---------------------------------------------------------------------------

PATCH_FORMAT_INSTRUCTIONS = """\
PATCH_FORMAT — changes only, do NOT output the complete file:
  Standard (any size):
    <<<<<<< SEARCH
    [exact lines to replace — copy verbatim from the code shown above, inside the fences]
    =======
    [replacement lines]
    >>>>>>> REPLACE
  Whole function/method:  REPLACE_FUNCTION: name  then ```lang\\n[full new body]\\n```
  Whole class:            REPLACE_CLASS: Name      then ```lang\\n[full new body]\\n```
  New code after entity:  INSERT_AFTER: name       then ```lang\\n[new block]\\n```
  Single line:            EDIT_LINE: N             then new content on the next line
  New file (no prior):    return the complete file as a code block — no patch needed
  Multiple edits: list patch blocks in order; all apply to the same file shown above."""

# ---------------------------------------------------------------------------
# Patch op dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SearchReplacePatch:
    search: str
    replace: str

@dataclass
class FunctionPatch:
    name: str
    new_body: str

@dataclass
class ClassPatch:
    name: str
    new_body: str

@dataclass
class InsertAfterPatch:
    anchor: str
    content: str

@dataclass
class EditLinePatch:
    line_number: int   # 1-indexed
    new_content: str

PatchOp = SearchReplacePatch | FunctionPatch | ClassPatch | InsertAfterPatch | EditLinePatch

# ---------------------------------------------------------------------------
# Fence helpers
# ---------------------------------------------------------------------------

def strip_code_fence(code_block: str) -> tuple[str, str]:
    """Remove opening/closing fence lines from a code block string.

    Returns (raw_content, lang_hint).  The block may include an optional
    filename comment as its first content line — that is kept intact inside
    the returned content; callers that want just the code should strip it
    themselves if needed.
    """
    lines = code_block.split("\n")
    lang = ""
    # strip opening fence
    if lines and _OPEN_FENCE.match(lines[0]):
        lang = lines[0].strip()[3:].strip()
        lines = lines[1:]
    # strip closing fence
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    # also strip a trailing empty line left by the fence removal
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines), lang


def _lang_from_key(artifact_key: str) -> str:
    """Derive a language hint from the artifact key (filename extension or fence lang)."""
    if artifact_key.startswith("__lang__:"):
        return artifact_key.split(":", 1)[1]
    if "." in artifact_key:
        ext = artifact_key.rsplit(".", 1)[-1].lower()
        return {
            "py": "python", "js": "javascript", "ts": "typescript",
            "tsx": "tsx", "jsx": "jsx", "html": "html", "css": "css",
            "rs": "rust", "go": "go", "java": "java", "cs": "csharp",
            "rb": "ruby", "php": "php", "swift": "swift", "kt": "kotlin",
            "c": "c", "cpp": "cpp", "h": "c", "hpp": "cpp",
            "sql": "sql", "sh": "bash", "bash": "bash", "md": "markdown",
        }.get(ext, ext)
    return ""

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_patch_response(text: str) -> list[PatchOp]:
    """Extract all patch operations from a model response, in document order."""
    ops: list[PatchOp] = []
    for m in _SEARCH_REPLACE_RE.finditer(text):
        ops.append(SearchReplacePatch(search=m.group(1), replace=m.group(2)))
    for m in _REPLACE_FUNC_RE.finditer(text):
        ops.append(FunctionPatch(name=m.group(1), new_body=m.group(2)))
    for m in _REPLACE_CLASS_RE.finditer(text):
        ops.append(ClassPatch(name=m.group(1), new_body=m.group(2)))
    for m in _INSERT_AFTER_RE.finditer(text):
        ops.append(InsertAfterPatch(anchor=m.group(1), content=m.group(2)))
    for m in _EDIT_LINE_RE.finditer(text):
        ops.append(EditLinePatch(line_number=int(m.group(1)), new_content=m.group(2)))
    return ops


def is_patch_response(text: str) -> bool:
    """Return True if the response contains at least one recognized patch marker."""
    return bool(_ANY_PATCH_MARKER.search(text))

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def _normalize_ws(text: str) -> str:
    """Normalize trailing whitespace and blank-line runs for fuzzy matching.
    Only used as a fallback when exact search fails — the stored content is
    always kept verbatim.
    """
    out, prev_blank = [], False
    for line in text.splitlines():
        s = line.rstrip()
        blank = not s
        if blank and prev_blank:
            continue
        out.append(s)
        prev_blank = blank
    return "\n".join(out)


def _apply_search_replace(text: str, op: SearchReplacePatch) -> tuple[str, bool]:
    # 1. Exact match
    if op.search in text:
        return text.replace(op.search, op.replace, 1), True
    # 2. Fuzzy fallback: normalize whitespace in both search and text
    norm_search = _normalize_ws(op.search)
    norm_text   = _normalize_ws(text)
    if norm_search in norm_text:
        # rebuild: find the original span whose normalized form matches
        for i in range(len(text)):
            candidate = text[i:i + len(op.search) + 20]
            if _normalize_ws(candidate).startswith(norm_search):
                end = i + len(candidate)
                # shrink end until normalized suffix no longer matches
                while end > i and _normalize_ws(text[i:end]) != norm_search:
                    end -= 1
                if _normalize_ws(text[i:end]) == norm_search:
                    return text[:i] + op.replace + text[end:], True
                break
    return text, False


def _apply_named_block(text: str, name: str, new_body: str) -> tuple[str, bool]:
    """Replace a named function/class block identified by block splitter."""
    from transformers import _split_definition_blocks, _clean_declaration_name
    blocks = _split_definition_blocks(text)
    if not blocks:
        return text, False
    target = name.lower()
    for header, block_text in blocks:
        if header is None:
            continue
        if _clean_declaration_name(header).lower() == target and block_text in text:
            return text.replace(block_text, new_body, 1), True
    return text, False


def _apply_insert_after(text: str, op: InsertAfterPatch) -> tuple[str, bool]:
    from transformers import _split_definition_blocks, _clean_declaration_name
    blocks = _split_definition_blocks(text)
    if not blocks:
        return text, False
    target = op.anchor.lower()
    for header, block_text in blocks:
        if header is None:
            continue
        if _clean_declaration_name(header).lower() == target and block_text in text:
            return text.replace(block_text, block_text.rstrip("\n") + "\n\n" + op.content, 1), True
    return text, False


def _apply_edit_line(text: str, op: EditLinePatch) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    idx = op.line_number - 1
    if not (0 <= idx < len(lines)):
        return text, False
    ending = "\n" if lines[idx].endswith("\n") else ""
    lines[idx] = op.new_content.rstrip("\n") + ending
    return "".join(lines), True


def apply_patch_ops(
    artifact_text: str, ops: list[PatchOp]
) -> tuple[str, int, int]:
    """Apply all patch ops to artifact_text in order.

    Returns (new_text, succeeded_count, failed_count).  Ops that fail leave
    the text unchanged at that step; successfully applied ops accumulate so
    the final text is as correct as possible even on partial failure.
    """
    text = artifact_text
    ok = failed = 0
    for op in ops:
        if isinstance(op, SearchReplacePatch):
            text, success = _apply_search_replace(text, op)
        elif isinstance(op, (FunctionPatch, ClassPatch)):
            text, success = _apply_named_block(text, op.name, op.new_body)
        elif isinstance(op, InsertAfterPatch):
            text, success = _apply_insert_after(text, op)
        elif isinstance(op, EditLinePatch):
            text, success = _apply_edit_line(text, op)
        else:
            success = False
        (ok if success else failed).__class__  # type: ignore[attr-defined]
        if success:
            ok += 1
        else:
            failed += 1
    return text, ok, failed

# ---------------------------------------------------------------------------
# Artifact resolution
# ---------------------------------------------------------------------------

def resolve_target_artifact(
    response_text: str,
    artifact_store: dict[str, str],
) -> str | None:
    """Choose which artifact the patches should be applied to.

    Resolution order (most- to least-confident):
    1. Filename mentioned explicitly in the response → match against store keys.
    2. Single artifact in the store → unambiguous.
    3. SEARCH text found verbatim in exactly one artifact → match by content.
    Returns the artifact key, or None if resolution fails.
    """
    if not artifact_store:
        return None

    resp_lower = response_text.lower()

    # 1. Filename mention (non-language-keyed artifacts only)
    for key in artifact_store:
        if not key.startswith("__lang__:") and key.lower() in resp_lower:
            return key

    # 2. Single artifact
    if len(artifact_store) == 1:
        return next(iter(artifact_store))

    # 3. SEARCH text found in exactly one artifact
    for m in _SEARCH_REPLACE_RE.finditer(response_text):
        search_text = m.group(1)
        matches = [k for k, v in artifact_store.items() if search_text in v]
        if len(matches) == 1:
            return matches[0]

    return None

# ---------------------------------------------------------------------------
# Response reconstruction (Option A — send full file to client)
# ---------------------------------------------------------------------------

def reconstruct_full_file_response(
    model_prose: str,
    patched_text: str,
    artifact_key: str,
    patches_applied: int,
    patches_failed: int,
) -> str:
    """Build the response the client will see after patch application.

    Keeps any explanatory prose the model wrote before the patch markers,
    then appends a summary line and the complete patched file wrapped in a
    code fence — exactly what the client (Cline, Continue, etc.) expects
    from a normal full-file generation turn.
    """
    lang = _lang_from_key(artifact_key)
    label = artifact_key if not artifact_key.startswith("__lang__:") else ""
    filename_comment = f"# {label}\n" if label else ""

    # Retain the model's prose up to the first patch marker
    prose_end = _ANY_PATCH_MARKER.search(model_prose)
    intro = model_prose[:prose_end.start()].rstrip() if prose_end else ""

    status_parts = [f"{patches_applied} patch{'es' if patches_applied != 1 else ''} applied"]
    if patches_failed:
        status_parts.append(f"{patches_failed} failed (kept previous content)")
    status_line = f"[ShapeShifter: {', '.join(status_parts)}] — complete updated file:"

    parts = []
    if intro:
        parts.append(intro)
    parts.append(status_line)
    parts.append(f"```{lang}\n{filename_comment}{patched_text}\n```")
    return "\n\n".join(parts)
