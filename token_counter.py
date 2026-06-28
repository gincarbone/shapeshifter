"""Token estimation — uses tiktoken when available, falls back to char/4."""
from __future__ import annotations

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text) // 4)


def compression_stats(before: str, after: str) -> dict:
    t_before = count_tokens(before)
    t_after  = count_tokens(after)
    ratio    = t_after / t_before if t_before else 1.0
    saved    = t_before - t_after
    return {
        "chars_before":  len(before),
        "chars_after":   len(after),
        "tokens_before": t_before,
        "tokens_after":  t_after,
        "tokens_saved":  saved,
        "compression_ratio": round(ratio, 4),
        "reduction_pct": round((1 - ratio) * 100, 1),
    }
