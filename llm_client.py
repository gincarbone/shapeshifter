"""Upstream LLM client — forwards requests to an OpenAI-compatible provider."""
from __future__ import annotations

import json
import time
from typing import AsyncGenerator

import httpx

_RESERVED_PARAMS = ("model", "messages", "temperature", "max_tokens", "stream")


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost/context-wrapper",
        "X-Title": "ShapeShifter",
    }


def _build_payload(
    model: str, messages: list[dict], temperature: float, max_tokens: int,
    extra_params: dict | None, stream: bool,
) -> dict:
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stream:
        payload["stream"] = True
    if extra_params:
        payload.update({k: v for k, v in extra_params.items() if k not in _RESERVED_PARAMS})
    return payload


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
    """Return (response_json, latency_ms). Non-streaming call."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = _build_payload(model, messages, temperature, max_tokens, extra_params, stream=False)

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=_headers(api_key), json=payload)
        r.raise_for_status()
    latency_ms = (time.monotonic() - t0) * 1000
    return r.json(), latency_ms


async def stream_upstream(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1200,
    extra_params: dict | None = None,
    timeout: float = 120.0,
) -> AsyncGenerator[dict, None]:
    """Stream chat-completion chunks from an OpenAI-compatible upstream as they
    arrive — no buffering of the full response. Yields each parsed SSE `data:`
    payload as a dict, in the same shape the upstream sent it (tool-call
    deltas, role/content deltas, and any trailing usage chunk are relayed
    unchanged). Stops on `data: [DONE]` or when the upstream closes the stream.

    Raises httpx.HTTPStatusError (with response body attached to the message)
    if the upstream rejects the request before any bytes are streamed back.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = _build_payload(model, messages, temperature, max_tokens, extra_params, stream=True)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, headers=_headers(api_key), json=payload) as r:
            if r.status_code >= 400:
                body = await r.aread()
                raise httpx.HTTPStatusError(
                    f"{r.status_code}: {body.decode(errors='replace')[:500]}",
                    request=r.request, response=r,
                )
            buffer = ""
            async for text_chunk in r.aiter_text():
                buffer += text_chunk
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.splitlines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            return
                        if not data:
                            continue
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue
