"""Auto mode selector - deterministic heuristics for MVP 0.1/0.2."""
from __future__ import annotations

import re


def _has_stacktrace(text: str) -> bool:
    return bool(re.search(r'Traceback|at\s+\w+\.\w+\(|caused by|stacktrace', text, re.IGNORECASE))


def _has_code(text: str) -> bool:
    return bool(re.search(r'```|def |class |function |import |#include|<\?php', text, re.IGNORECASE))


def _has_error(text: str) -> bool:
    return bool(re.search(r'ERROR|Exception|FATAL|SyntaxError|NullPointer|undefined', text, re.IGNORECASE))


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    return (stripped.startswith('{') and stripped.endswith('}')) or \
           (stripped.startswith('[') and stripped.endswith(']'))


def _is_comparison(text: str) -> bool:
    return bool(re.search(r'compare|comparison|difference|vs\.?|versus|table', text, re.IGNORECASE))


def _is_email_or_text(text: str) -> bool:
    return bool(re.search(r'email|letter|message|text|write\s+a\b', text, re.IGNORECASE))


def choose_mode(context: str, user_request: str) -> str:
    combined = context + " " + user_request

    if _has_stacktrace(combined):
        return "matrix"

    if _has_code(combined) and _has_error(combined):
        return "hybrid"

    if _looks_like_json(context):
        return "json"

    if _is_comparison(user_request):
        return "table"

    if _is_email_or_text(user_request):
        return "minimal"

    if len(context) < 2000:
        return "minimal"

    return "hybrid"
