"""Upstream LLM client — forwards requests to an OpenAI-compatible provider."""
from __future__ import annotations

import time
import httpx


async def call_upstream(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1200,
    extra_params: dict | None = None,
    timeout: float = 120.0,
) -> tuple[dict, float]:
    """Return (response_json, latency_ms)."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost/context-wrapper",
        "X-Title": "ShapeShifter",
    }
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_params:
        payload.update({k: v for k, v in extra_params.items()
                        if k not in ("model", "messages", "temperature", "max_tokens")})

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
    latency_ms = (time.monotonic() - t0) * 1000
    return r.json(), latency_ms
