# Copyright (c) 2026 Marcello Incarbone. MIT License — see LICENSE file.
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


def _is_coding_session(context: str) -> bool:
    """True when the context contains a multi-turn coding exchange in any language.

    Detection is language-agnostic: we look for fenced code blocks (``` with any
    tag or none) or universal programming constructs rather than listing every
    language explicitly.
    """
    if not re.search(r'\[USER\]|\[ASSISTANT\]', context):
        return False

    # Any fenced code block (``` optionally followed by a language name)
    if re.search(r'```\w*\s*\n', context):
        return True

    # Universal programming constructs present in virtually every language
    code_signals = re.compile(
        r'(?:'
        # Function / method definitions (parens optional — e.g. Ruby `def greet`)
        r'def\s+\w+[\s(]|function\s+\w+\s*\(|fn\s+\w+\s*\('
        r'|fun\s+\w+\s*\(|func\s+\w+\s*\(|sub\s+\w+\s*\(|proc\s+\w+\s*\('
        # Class / struct / interface declarations
        r'|class\s+\w+|struct\s+\w+|interface\s+\w+|enum\s+\w+'
        r'|impl\s+\w+|trait\s+\w+'
        # Import / module statements
        r'|import\s+\w+|from\s+\w+\s+import|require\s*\(|include\s+[<"]'
        r'|use\s+\w+::|using\s+\w+|extern\s+crate'
        # Common declarations and operators
        r'|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*='
        r'|#include|#define|package\s+\w+|namespace\s+\w+'
        # Markup / template roots
        r'|<!DOCTYPE|<html\b|<\?php|<\?xml'
        r')',
        re.IGNORECASE,
    )
    return bool(code_signals.search(context))


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

def transform_raw(context: str) -> str:
    return context


def transform_minimal(context: str) -> str:
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


def transform_yaml(context: str) -> str:
    # For multi-turn coding sessions: emit requirements as yaml list, skip code blobs
    if _is_coding_session(context):
        reqs = _extract_user_requirements(context)
        task = _infer_task(context)
        data: dict = {
            "context_mode": "yaml",
            "task": task,
            "cumulative_requirements": reqs,
            "constraint": "Return COMPLETE file. All requirements must be present.",
        }
        return "context_mode: yaml\n\n" + yaml.dump(data, default_flow_style=False, allow_unicode=True).strip()

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


def transform_json(context: str) -> str:
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


def transform_table(context: str) -> str:
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


def transform_hybrid(context: str) -> str:
    # For multi-turn coding sessions: structured requirements list + tech stack
    if _is_coding_session(context):
        reqs = _extract_user_requirements(context)
        stack = _infer_stack(context)
        parts = [
            "context_mode: hybrid",
            "",
            f"task: generation",
            f"  constraint: return COMPLETE file, all requirements must be present",
            "",
            "cumulative_requirements:",
        ]
        for i, r in enumerate(reqs, 1):
            parts.append(f"  [{i}]: {r}")
        if stack:
            parts += ["", "environment:", "  stack: " + ", ".join(stack)]
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


def transform_symbolic(context: str) -> str:
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


def transform_incremental(context: str) -> str:
    """For multi-turn coding/generation sessions.

    Keeps all USER requirements across turns (the feature list the model must
    honour) and discards ASSISTANT responses (the code). The model can regenerate
    code from requirements but cannot recover requirements that were lost.
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

    lines = [
        "CODING_SESSION: incremental",
        "RULE: Every requirement listed below MUST be present in the final output.",
        "",
        "REQUIREMENTS (in order — all cumulative):",
    ]
    for i, req in enumerate(user_reqs, 1):
        lines.append(f"\n[{i}] {req}")

    lines += [
        "",
        "CONSTRAINT: Return the COMPLETE updated file. No truncation. No placeholders.",
    ]
    return "\n".join(lines)


def transform_matrix(context: str) -> str:
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

TRANSFORMERS: dict[str, Callable[[str], str]] = {
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


def apply_transform(mode: str, messages: list[dict]) -> tuple[str, str]:
    """Return (original_context, transformed_context)."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown context mode: {mode!r}. Valid: {sorted(VALID_MODES)}")

    raw = "\n\n".join(
        f"[{m['role'].upper()}]\n{m.get('content', '')}"
        for m in messages
        if isinstance(m.get("content"), str)
    )
    transformed = TRANSFORMERS[mode](raw)
    return raw, transformed
