# Copyright (c) 2026 Gaetano Marcello Incarbone. MIT License — see LICENSE file.
"""ShapeShifter — local OpenAI-compatible context compression proxy."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from llm_client import call_upstream, stream_upstream
from mode_selector import choose_mode
from output_contracts import build_system_prompt, detect_contract_type
from patch_engine import (
    apply_patch_ops, is_patch_response, parse_patch_response,
    reconstruct_full_file_response, resolve_target_artifact, strip_code_fence,
)
from token_counter import compression_stats, count_tokens
from transformers import VALID_MODES, apply_transform, _extract_retrievable_pieces

from alfa1_routes import router as alfa1_router

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Placeholder shipped in .env.example — never treat this as a real key.
_PLACEHOLDER_KEY  = "your-api-key-here"

HOST              = os.getenv("WRAPPER_HOST", "127.0.0.1")
PORT              = int(os.getenv("WRAPPER_PORT", "8787"))
UPSTREAM_URL      = os.getenv("UPSTREAM_BASE_URL", "")
UPSTREAM_KEY      = os.getenv("UPSTREAM_API_KEY", "")
if UPSTREAM_KEY == _PLACEHOLDER_KEY:
    UPSTREAM_KEY = ""
DEFAULT_MODEL     = os.getenv("DEFAULT_MODEL", "deepseek/deepseek-chat")
CONTEXT_MODE      = os.getenv("CONTEXT_MODE", "hybrid")
AUTO_MODE         = os.getenv("AUTO_MODE", "false").lower() == "true"
LOG_REQUESTS      = os.getenv("LOG_REQUESTS", "true").lower() == "true"
LOG_RESPONSES     = os.getenv("LOG_RESPONSES", "true").lower() == "true"
LOG_DIR           = Path(os.getenv("LOG_DIR", "logs"))
MAX_OUTPUT_TOKENS = int(os.getenv("DEFAULT_MAX_OUTPUT_TOKENS", "1200"))

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Live stats (in-memory, reset on restart)
# ---------------------------------------------------------------------------

_start_time = time.monotonic()
_start_wall  = datetime.now(timezone.utc)

_stats: dict = {
    "total_requests":           0,
    "total_tokens_before":      0,
    "total_tokens_after":       0,
    "total_tokens_saved":       0,
    "total_output_tokens_saved": 0,
    "by_mode": {m: {"count": 0, "tok_before": 0, "tok_after": 0,
                    "tok_saved": 0, "out_tok_saved": 0}
                for m in VALID_MODES},
}
_recent: deque[dict] = deque(maxlen=50)
_ctx_store: dict[str, dict] = {}   # request_id -> {raw, transformed} (last 50)
_ctx_store_keys: deque[str] = deque(maxlen=50)
_sse_queues: list[asyncio.Queue] = []
_current_model: str = DEFAULT_MODEL
_model_input_cost_per_1m: float | None = None   # $/1M input tokens, None if unknown
_model_output_cost_per_1m: float | None = None  # $/1M output tokens, None if unknown

# Mutable runtime config (mirrors .env, survives within session, persisted on save)
_config: dict = {
    "upstream_base_url": UPSTREAM_URL,
    "upstream_api_key":  UPSTREAM_KEY,
    "default_model":     DEFAULT_MODEL,
    "context_mode":      CONTEXT_MODE,
    "auto_mode":         AUTO_MODE,
    "log_requests":      LOG_REQUESTS,
    "log_responses":     LOG_RESPONSES,
    "log_dir":           str(LOG_DIR),
}

_ENV_PATH  = Path(__file__).parent / ".env"
_KEYS_PATH = Path(__file__).parent / ".shapeshifter_keys.json"

# Known OpenAI-compatible providers
KNOWN_PROVIDERS: list[dict] = [
    {"name": "OpenRouter",   "url": "https://openrouter.ai/api/v1",     "key_required": True},
    {"name": "DeepSeek",     "url": "https://api.deepseek.com/v1",       "key_required": True},
    {"name": "OpenAI",       "url": "https://api.openai.com/v1",         "key_required": True},
    {"name": "Groq",         "url": "https://api.groq.com/openai/v1",    "key_required": True},
    {"name": "Together AI",  "url": "https://api.together.xyz/v1",       "key_required": True},
    {"name": "Mistral",      "url": "https://api.mistral.ai/v1",         "key_required": True},
    {"name": "Fireworks",    "url": "https://api.fireworks.ai/inference/v1", "key_required": True},
    {"name": "Ollama",       "url": "http://localhost:11434/v1",          "key_required": False},
    {"name": "LM Studio",    "url": "http://localhost:1234/v1",           "key_required": False},
]
_LOCAL_URLS = {p["url"] for p in KNOWN_PROVIDERS if not p["key_required"]}

# Per-provider API key storage (url → key), persisted to .shapeshifter_keys.json
_provider_keys: dict[str, str] = {}


def _load_provider_keys() -> None:
    global _provider_keys
    if _KEYS_PATH.exists():
        try:
            _provider_keys = json.loads(_KEYS_PATH.read_text(encoding="utf-8"))
        except Exception:
            _provider_keys = {}
    # seed from current .env key if not already stored
    if UPSTREAM_URL and UPSTREAM_KEY and UPSTREAM_URL not in _provider_keys:
        _provider_keys[UPSTREAM_URL] = UPSTREAM_KEY


def _save_provider_keys() -> None:
    _KEYS_PATH.write_text(json.dumps(_provider_keys, indent=2, ensure_ascii=False), encoding="utf-8")


_load_provider_keys()

# Adopt a previously-saved key for the active provider if .env didn't supply
# one (e.g. .env still has the placeholder, but a real key was saved earlier
# via the dashboard and persisted to .shapeshifter_keys.json).
if UPSTREAM_URL and not UPSTREAM_KEY and UPSTREAM_URL in _provider_keys:
    UPSTREAM_KEY = _provider_keys[UPSTREAM_URL]
    _config["upstream_api_key"] = UPSTREAM_KEY


def _key_status(url: str) -> str:
    """Return 'saved', 'not_set', or 'not_required'."""
    if url in _LOCAL_URLS:
        return "not_required"
    return "saved" if _provider_keys.get(url) else "not_set"


def _mask_key(key: str) -> str:
    if not key:
        return ""
    return key[:8] + "..." + key[-4:] if len(key) > 12 else "*" * len(key)

_ENV_KEY_MAP = {
    "upstream_base_url": "UPSTREAM_BASE_URL",
    "upstream_api_key":  "UPSTREAM_API_KEY",
    "default_model":     "DEFAULT_MODEL",
    "context_mode":      "CONTEXT_MODE",
    "auto_mode":         "AUTO_MODE",
    "log_requests":      "LOG_REQUESTS",
    "log_responses":     "LOG_RESPONSES",
    "log_dir":           "LOG_DIR",
}


def _apply_config(cfg: dict) -> None:
    """Apply a config dict to runtime globals."""
    global UPSTREAM_URL, UPSTREAM_KEY, _current_model, CONTEXT_MODE
    global AUTO_MODE, LOG_REQUESTS, LOG_RESPONSES, LOG_DIR
    if "upstream_base_url" in cfg:
        UPSTREAM_URL = cfg["upstream_base_url"]
        # auto-load saved key for this URL if no explicit key provided
        if "upstream_api_key" not in cfg and UPSTREAM_URL in _provider_keys:
            UPSTREAM_KEY = _provider_keys[UPSTREAM_URL]
            _config["upstream_api_key"] = UPSTREAM_KEY
    if "upstream_api_key" in cfg:
        UPSTREAM_KEY = cfg["upstream_api_key"]
        # persist key for current URL
        if UPSTREAM_URL and UPSTREAM_KEY:
            _provider_keys[UPSTREAM_URL] = UPSTREAM_KEY
            _save_provider_keys()
    if "default_model" in cfg:
        _current_model = cfg["default_model"]
    if "context_mode" in cfg and cfg["context_mode"] in VALID_MODES:
        CONTEXT_MODE = cfg["context_mode"]
    if "auto_mode" in cfg:
        AUTO_MODE = str(cfg["auto_mode"]).lower() in ("true", "1", "yes")
    if "log_requests" in cfg:
        LOG_REQUESTS = str(cfg["log_requests"]).lower() in ("true", "1", "yes")
    if "log_responses" in cfg:
        LOG_RESPONSES = str(cfg["log_responses"]).lower() in ("true", "1", "yes")
    if "log_dir" in cfg:
        LOG_DIR = Path(cfg["log_dir"])
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    _config.update(cfg)


def _persist_env(cfg: dict) -> None:
    """Write changed keys back to .env, preserving comments and unknown lines."""
    lines = _ENV_PATH.read_text(encoding="utf-8").splitlines() if _ENV_PATH.exists() else []
    updated: set[str] = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        if "=" in stripped:
            env_key = stripped.split("=", 1)[0].strip()
            for cfg_key, mapped in _ENV_KEY_MAP.items():
                if env_key == mapped and cfg_key in cfg:
                    val = cfg[cfg_key]
                    if isinstance(val, bool):
                        val = "true" if val else "false"
                    new_lines.append(f"{mapped}={val}")
                    updated.add(cfg_key)
                    break
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    # append any keys not already in file
    for cfg_key, val in cfg.items():
        if cfg_key not in updated and cfg_key in _ENV_KEY_MAP:
            if isinstance(val, bool):
                val = "true" if val else "false"
            new_lines.append(f"{_ENV_KEY_MAP[cfg_key]}={val}")
    _ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _record_stats(
    mode: str, s: dict, latency_ms: float,
    request_id: str = "", model: str = "", output_tokens_saved: int = 0,
) -> None:
    _stats["total_requests"]            += 1
    _stats["total_tokens_before"]       += s["tokens_before"]
    _stats["total_tokens_after"]        += s["tokens_after"]
    _stats["total_tokens_saved"]        += s["tokens_saved"]
    _stats["total_output_tokens_saved"] += output_tokens_saved
    bm = _stats["by_mode"].setdefault(
        mode,
        {"count": 0, "tok_before": 0, "tok_after": 0, "tok_saved": 0, "out_tok_saved": 0},
    )
    bm["count"]        += 1
    bm["tok_before"]   += s["tokens_before"]
    bm["tok_after"]    += s["tokens_after"]
    bm["tok_saved"]    += s["tokens_saved"]
    bm["out_tok_saved"] += output_tokens_saved

    entry = {
        "ts":                 datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "request_id":         request_id,
        "mode":               mode,
        "model":              model,
        "tok_before":         s["tokens_before"],
        "tok_after":          s["tokens_after"],
        "tok_saved":          s["tokens_saved"],
        "reduction_pct":      s["reduction_pct"],
        "latency_ms":         round(latency_ms, 0),
        "out_tok_saved":      output_tokens_saved,
    }
    _recent.appendleft(entry)

    event = json.dumps({"stats": _build_summary(), "latest": entry})
    for q in list(_sse_queues):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


def _build_summary() -> dict:
    uptime_s = int(time.monotonic() - _start_time)
    avg_ratio = (
        _stats["total_tokens_after"] / _stats["total_tokens_before"]
        if _stats["total_tokens_before"] else 1.0
    )
    by_mode_out = {}
    for m, v in _stats["by_mode"].items():
        if v["count"] == 0:
            continue
        by_mode_out[m] = {
            "count":       v["count"],
            "avg_before":  round(v["tok_before"] / v["count"]),
            "avg_after":   round(v["tok_after"] / v["count"]),
            "avg_saved":   round(v["tok_saved"] / v["count"]),
            "avg_reduction_pct": round(
                (1 - v["tok_after"] / v["tok_before"]) * 100 if v["tok_before"] else 0, 1
            ),
        }
    dollars_saved = None
    if _model_input_cost_per_1m is not None and _stats["total_tokens_saved"] > 0:
        dollars_saved = round(_stats["total_tokens_saved"] / 1_000_000 * _model_input_cost_per_1m, 6)

    dollars_saved_output = None
    if _model_output_cost_per_1m is not None and _stats["total_output_tokens_saved"] > 0:
        dollars_saved_output = round(
            _stats["total_output_tokens_saved"] / 1_000_000 * _model_output_cost_per_1m, 6
        )

    return {
        "total_requests":            _stats["total_requests"],
        "total_tokens_saved":        _stats["total_tokens_saved"],
        "total_tokens_before":       _stats["total_tokens_before"],
        "total_output_tokens_saved": _stats["total_output_tokens_saved"],
        "avg_ratio":                 round(avg_ratio, 3),
        "avg_reduction_pct":         round((1 - avg_ratio) * 100, 1),
        "uptime_s":                  uptime_s,
        "by_mode":                   by_mode_out,
        "dollars_saved":             dollars_saved,
        "dollars_saved_output":      dollars_saved_output,
        "model_input_cost_per_1m":   _model_input_cost_per_1m,
        "model_output_cost_per_1m":  _model_output_cost_per_1m,
        "current_model":             _current_model,
    }

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ShapeShifter", version="0.2.0")
app.include_router(alfa1_router, prefix="/alfa1")


def _log(filename: str, record: dict) -> None:
    path = LOG_DIR / filename
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _content_as_str(msg: dict) -> str:
    """Return message content as plain string.
    Handles both string content and multimodal list content
    (e.g. Cline's [{"type":"text","text":"..."},...] format).
    """
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _is_agentic(messages: list[dict], body: dict | None = None) -> bool:
    """True for tool-calling / function-calling exchanges — OpenAI-style
    (assistant `tool_calls`, `role: tool`/`function`) or Anthropic-style
    content blocks (`tool_use` / `tool_result`), or a request that declares
    `tools`/`tool_choice`. These must pass through uncompressed with their
    original structure intact, or the model loses the ability to correlate
    tool calls with their results.
    """
    if body and (body.get("tools") or body.get("tool_choice")):
        return True
    for m in messages:
        if m.get("role") in ("tool", "function"):
            return True
        if m.get("tool_calls") or m.get("function_call"):
            return True
        content = m.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("tool_use", "tool_result"):
                    return True
    return False


_FILE_READ_TOOL_NAME = re.compile(r"read.*file|get.*file.*content|cat_file|file_read", re.IGNORECASE)
_FILE_PATH_ARG_KEYS = ("path", "file_path", "filepath", "file", "filename", "target_file")


def _build_tool_call_keys(messages: list[dict]) -> dict[str, tuple[str, str]]:
    """Map tool_call_id -> (dedup_key, human_label) for EVERY tool call, not
    just file reads (Feature 7 — generalizes Feature 1's file-read-only
    scope). A repeated `execute_command("npm test")` or `search("TODO")`
    with identical arguments is exactly as safe to dedupe as a repeated
    `read_file("app.py")`: nothing is lost, the full result still exists
    later in the same request.

    File-read-like calls (matched by function name, with a resolvable path
    argument) get a clean filename-based key/label for readability, matching
    the original behavior. Every other tool call gets a general key built
    from (function name, canonicalized arguments) and a label using just the
    function name.
    """
    result: dict[str, tuple[str, str]] = {}
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw)
            except (ValueError, TypeError):
                args = None

            path = None
            if _FILE_READ_TOOL_NAME.search(name) and isinstance(args, dict):
                for key in _FILE_PATH_ARG_KEYS:
                    if isinstance(args.get(key), str):
                        path = args[key]
                        break

            if path:
                result[tc.get("id", "")] = (f"file:{path}", path)
            else:
                canonical_args = json.dumps(args, sort_keys=True) if isinstance(args, dict) else args_raw
                result[tc.get("id", "")] = (f"call:{name}:{canonical_args}", name)
    return result


def _dedupe_repeated_tool_calls(messages: list[dict]) -> list[dict]:
    """Within a single agentic request, if ANY tool call is repeated later
    with the exact same (name, arguments) — not just file reads — keep only
    the LAST occurrence of that call's result in full and replace every
    earlier one with a short marker, whether or not the result actually
    changed between calls. This mirrors the same "latest wins" retention
    already applied to assistant/user code in transformers.py
    (`_extract_latest_artifacts`): only the current result of a repeated
    call is needed to act on it now.

    Messages that aren't `tool` role are left completely untouched. The
    size guard still applies: a marker never replaces content it isn't
    actually shorter than.
    """
    tool_keys = _build_tool_call_keys(messages)
    if not tool_keys:
        return messages

    last_index_for_key: dict[str, int] = {}
    for i, m in enumerate(messages):
        if m.get("role") != "tool" or not isinstance(m.get("content"), str):
            continue
        entry = tool_keys.get(m.get("tool_call_id", ""))
        if entry:
            last_index_for_key[entry[0]] = i

    new_messages: list[dict] = []
    for i, m in enumerate(messages):
        if m.get("role") != "tool" or not isinstance(m.get("content"), str):
            new_messages.append(m)
            continue
        entry = tool_keys.get(m.get("tool_call_id", ""))
        if entry and last_index_for_key.get(entry[0]) != i:
            key, label = entry
            unchanged = m["content"] == messages[last_index_for_key[key]].get("content")
            if key.startswith("file:"):
                marker = (
                    f"[{label} unchanged since earlier read — content omitted]" if unchanged
                    else f"[{label} read here — since superseded by a later version shown further below]"
                )
            else:
                marker = (
                    f"[{label} call repeated with identical arguments — output unchanged, "
                    f"see the repeated call's result later in this conversation]" if unchanged
                    else f"[{label} call repeated with identical arguments — since superseded "
                         f"by a later result shown further below]"
                )
            # Never replace with something that isn't actually smaller — for a
            # small result the marker itself can outweigh the content it'd replace.
            if len(marker) < len(m["content"]):
                new_messages.append({**m, "content": marker})
                continue
        new_messages.append(m)
    return new_messages


def _resolve_mode(request_data: dict, http_headers: dict) -> str:
    mode = http_headers.get("x-context-mode", "").strip().lower()
    if mode and mode in VALID_MODES:
        return mode
    mode = request_data.get("context_mode", "").strip().lower()
    if mode and mode in VALID_MODES:
        return mode
    if not AUTO_MODE and CONTEXT_MODE in VALID_MODES:
        return CONTEXT_MODE
    if AUTO_MODE:
        messages = request_data.get("messages", [])
        raw_ctx  = " ".join(_content_as_str(m) for m in messages if _content_as_str(m))
        user_req = next(
            (_content_as_str(m) for m in reversed(messages) if m.get("role") == "user"),
            ""
        )
        return choose_mode(raw_ctx, user_req)
    return "hybrid"


def _build_compressed_messages(
    original_messages: list[dict], mode: str
) -> tuple[list[dict], str, str, dict[str, str]]:
    """Compress conversation history for non-agentic (plain chat) requests.

    Only messages before the last user turn are compressed. Everything from
    the last user turn onward is forwarded verbatim — not just that single
    message — so pasted code, file contents, or any trailing messages are
    never stripped. A client-supplied system message (e.g. a client's own
    behavioral prompt) is preserved verbatim and combined with ShapeShifter's
    own directive rather than being discarded.

    The 4th return value is a retrieval map (key -> full content) for
    everything Feature 2/3 may have collapsed in this turn's compressed
    history — used by the retrieval tool (Feature 6) to answer the model's
    own `shapeshifter_expand` calls instantly. Empty for modes that don't do
    artifact retention.
    """
    client_system = next((m for m in original_messages if m.get("role") == "system"), None)
    non_system    = [m for m in original_messages if m.get("role") != "system"]

    last_user_idx = next(
        (i for i in range(len(non_system) - 1, -1, -1)
         if non_system[i].get("role") == "user"),
        None,
    )
    if last_user_idx is not None and last_user_idx > 0:
        history = non_system[:last_user_idx]
        current = non_system[last_user_idx:]
    else:
        history = []
        current = non_system[-1:] if non_system else [{"role": "user", "content": ""}]

    current_text = "\n\n".join(m.get("content", "") for m in current if isinstance(m.get("content"), str))
    raw_ctx, transformed_ctx = apply_transform(mode, history, current_text) if history else ("", "")
    # Contract type is frozen to the FIRST user turn, not re-derived from the
    # whole (growing) history: re-scanning every turn lets a later keyword
    # flip the OUTPUT_CONTRACT section of the system message mid-session,
    # which would break the byte-stable prefix providers rely on for prompt
    # caching (see Feature 4, docs/token_savings_roadmap.md). The opening
    # ask defines the task type; later incidental keywords shouldn't.
    first_user_msg = next((m for m in original_messages if m.get("role") == "user"), None)
    contract_type    = detect_contract_type([first_user_msg] if first_user_msg else [])
    shapeshifter_sys = build_system_prompt(mode, contract_type)
    client_sys_text  = client_system.get("content", "") if client_system else ""
    system_content = (
        f"{client_sys_text}\n\n{shapeshifter_sys}"
        if isinstance(client_sys_text, str) and client_sys_text.strip()
        else shapeshifter_sys
    )

    new_messages: list[dict] = [{"role": "system", "content": system_content}]
    if transformed_ctx:
        new_messages.append({"role": "user",      "content": transformed_ctx})
        new_messages.append({"role": "assistant",  "content": "Understood."})
    new_messages.extend(current)

    # stats are computed over the full raw context vs compressed history
    full_raw = (
        apply_transform("raw", non_system)[0] if history
        else "\n\n".join(m.get("content", "") for m in current if isinstance(m.get("content"), str))
    )
    # Only the coding-session modes ever produce a collapsed stub in the
    # first place — computing this for other modes would inject a tool the
    # model could never usefully call.
    retrieval_map = _extract_retrievable_pieces(raw_ctx) if history and mode in ("hybrid", "yaml", "incremental") else {}
    return new_messages, full_raw, transformed_ctx, retrieval_map

# ---------------------------------------------------------------------------
# Output patch processing (Option A — reconstruct full file for client)
# ---------------------------------------------------------------------------

def _build_raw_artifact_store(retrieval_map: dict[str, str]) -> dict[str, str]:
    """Build a store of {artifact_key: raw_content} from the retrieval map.

    `retrieval_map` values are fenced code blocks (```lang\\n...\\n```).
    Patch application operates on the raw text inside the fences, so we strip
    them here once and reuse the result for both resolution and application.
    Only whole-file entries are included (per-function sub-keys like
    "calc.py#divide" are excluded — patching targets full files).
    """
    store: dict[str, str] = {}
    for key, value in retrieval_map.items():
        if "#" not in key:   # skip per-function sub-keys
            raw, _ = strip_code_fence(value)
            store[key] = raw
    return store


def _process_patch_response(
    output_text: str, retrieval_map: dict[str, str]
) -> tuple[str, int, int, int]:
    """Detect, apply, and reconstruct a patch response into a full file.

    Returns (final_output_text, output_tokens_saved, patches_applied, patches_failed).
    If the response is not a patch, returns the original text with zeros.

    Steps:
    1. Quick check: does the response contain any patch markers?
    2. Strip fences from artifacts in retrieval_map to get raw content.
    3. Resolve which artifact is the target.
    4. Parse + apply all patch ops in order.
    5. Reconstruct a full-file response the client can use directly.
    6. Compute output token savings: (full file size) - (patch text size).
    """
    if not retrieval_map or not is_patch_response(output_text):
        return output_text, 0, 0, 0

    raw_store = _build_raw_artifact_store(retrieval_map)
    target_key = resolve_target_artifact(output_text, raw_store)
    if not target_key:
        return output_text, 0, 0, 0

    artifact_raw = raw_store[target_key]
    ops = parse_patch_response(output_text)
    if not ops:
        return output_text, 0, 0, 0

    patched_text, ok, failed = apply_patch_ops(artifact_raw, ops)

    # Tokens: what the model actually produced vs what a full file would cost
    patch_tokens    = count_tokens(output_text)
    fullfile_tokens = count_tokens(artifact_raw)
    saved = max(0, fullfile_tokens - patch_tokens)

    reconstructed = reconstruct_full_file_response(
        model_prose=output_text,
        patched_text=patched_text,
        artifact_key=target_key,
        patches_applied=ok,
        patches_failed=failed,
    )
    return reconstructed, saved, ok, failed


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0", "uptime_s": int(time.monotonic() - _start_time)}


@app.get("/v1/config/model")
async def get_model():
    return {"model": _current_model,
            "input_cost_per_1m": _model_input_cost_per_1m,
            "output_cost_per_1m": _model_output_cost_per_1m}


@app.post("/v1/config/model")
async def set_model(request: Request):
    global _current_model, _model_input_cost_per_1m, _model_output_cost_per_1m
    try:
        body = await request.json()
        new_model = body.get("model", "").strip()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if not new_model:
        return JSONResponse({"error": "model field required"}, status_code=400)
    _current_model = new_model
    _config["default_model"] = new_model
    # optional pricing from Browse selection
    in_cost = body.get("input_cost_per_1m")
    if in_cost is not None:
        try:
            _model_input_cost_per_1m = float(in_cost)
        except (TypeError, ValueError):
            _model_input_cost_per_1m = None
    out_cost = body.get("output_cost_per_1m")
    if out_cost is not None:
        try:
            _model_output_cost_per_1m = float(out_cost)
        except (TypeError, ValueError):
            _model_output_cost_per_1m = None
    return {"model": _current_model, "status": "updated",
            "input_cost_per_1m": _model_input_cost_per_1m,
            "output_cost_per_1m": _model_output_cost_per_1m}


def _is_ollama_url(base: str) -> bool:
    return ":11434" in base


def _normalise_models(raw: dict) -> list[dict]:
    """Convert any provider's model list response to our standard format."""
    models = []

    # OpenAI-compatible: {"data": [...]}
    items = raw.get("data") or []

    # Ollama /api/tags: {"models": [...]}
    if not items and "models" in raw:
        for m in raw["models"]:
            name = m.get("name") or m.get("model", "")
            details = m.get("details") or {}
            label = name
            if details.get("parameter_size"):
                label += f" ({details['parameter_size']}"
                if details.get("quantization_level"):
                    label += f" {details['quantization_level']}"
                label += ")"
            models.append({
                "id": name, "name": label,
                "context_length": None,
                "input_cost_per_1m": None,
                "output_cost_per_1m": None,
            })
        return sorted(models, key=lambda x: x["id"])

    for m in items:
        pricing = m.get("pricing") or {}

        # OpenRouter: pricing.prompt / pricing.completion (per-token strings)
        def _cost(key: str) -> float | None:
            try:
                v = float(pricing.get(key, 0) or 0)
                return round(v * 1_000_000, 4) if v else (0.0 if key in pricing else None)
            except (TypeError, ValueError):
                return None

        inp = _cost("prompt")
        out = _cost("completion")

        # Together AI: pricing.input / pricing.output (per-million floats)
        if inp is None and "input" in pricing:
            try:   inp = round(float(pricing["input"]),  4)
            except Exception: pass
        if out is None and "output" in pricing:
            try:   out = round(float(pricing["output"]), 4)
            except Exception: pass

        models.append({
            "id":              m.get("id", ""),
            "name":            m.get("name") or m.get("id", ""),
            "context_length":  m.get("context_length") or m.get("context_window"),
            "input_cost_per_1m":  inp,
            "output_cost_per_1m": out,
        })

    return sorted(models, key=lambda x: x["id"])


async def _fetch_provider_models(target_base: str, api_key: str) -> list[dict]:
    """Fetch + normalise the model list for a given provider base URL."""
    import httpx  # already installed via requirements

    headers: dict = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Ollama uses /api/tags instead of /v1/models
    if _is_ollama_url(target_base):
        # strip /v1 suffix if present to reach the root
        ollama_root = target_base.replace("/v1", "")
        fetch_url = ollama_root.rstrip("/") + "/api/tags"
    else:
        fetch_url = target_base + "/models"

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(fetch_url, headers=headers)
        r.raise_for_status()
        raw = r.json()

    return _normalise_models(raw)


@app.get("/v1/upstream/models")
async def upstream_models(url: str = ""):
    """Proxy GET /models to the requested provider (or active upstream if url omitted).

    The caller passes ?url=<base_url> so Browse always fetches from the provider
    currently selected in the dashboard, not necessarily the active one.
    Key is looked up from _provider_keys; falls back to UPSTREAM_KEY for active URL.
    """
    target_base = (url.strip() or UPSTREAM_URL).rstrip("/")
    if not target_base:
        return JSONResponse({"error": "No provider URL specified or configured"}, status_code=503)

    # resolve key: use saved key for that URL, fall back to active key if same URL
    api_key = _provider_keys.get(target_base) or (UPSTREAM_KEY if target_base == UPSTREAM_URL.rstrip("/") else "")

    try:
        models = await _fetch_provider_models(target_base, api_key)
    except Exception as exc:
        return JSONResponse({"error": f"Upstream error: {exc}"}, status_code=502)

    return JSONResponse({"data": models, "count": len(models), "provider_url": target_base})


@app.get("/v1/models")
async def list_models():
    """Standard OpenAI models-list endpoint. OpenAI-compatible clients (Cline,
    Continue, etc.) call this to discover available models and their real
    context length — without it, a client has no way to know e.g. that
    `deepseek/deepseek-v4-flash` has a 1M-token context and silently falls
    back to a generic default (128K is a common one), which shows up in the
    client's own context-usage UI as an artificially small window. This has
    nothing to do with ShapeShifter's compression — it's purely a discovery
    gap this endpoint closes by proxying the active upstream's real model
    list, reusing the same fetch/normalise path as the dashboard's Browse
    feature, in the standard `{"object": "list", "data": [...]}` envelope
    with `context_length` included as the (widely supported, if unofficial)
    extra field clients look for.
    """
    target_base = UPSTREAM_URL.rstrip("/")
    if not target_base:
        return JSONResponse({"object": "list", "data": []})

    api_key = _provider_keys.get(target_base) or UPSTREAM_KEY
    try:
        models = await _fetch_provider_models(target_base, api_key)
    except Exception:
        # Upstream unreachable — still respond in the standard shape (with
        # just the configured default model, no metadata) rather than
        # erroring out a client that only wanted a model list to render.
        models = [{"id": _current_model, "context_length": None}]

    data = [{
        "id": m["id"],
        "object": "model",
        "created": 0,
        "owned_by": m["id"].split("/", 1)[0] if "/" in m["id"] else "shapeshifter",
        "context_length": m.get("context_length"),
    } for m in models]
    return JSONResponse({"object": "list", "data": data})


async def _auto_resolve_model_pricing() -> None:
    """Best-effort: look up pricing for the active model so the 'Est. $ Saved'
    card is populated on startup without requiring a manual Browse selection."""
    global _model_input_cost_per_1m, _model_output_cost_per_1m
    target_base = UPSTREAM_URL.rstrip("/")
    if not target_base or not _current_model:
        return
    api_key = _provider_keys.get(target_base) or UPSTREAM_KEY
    try:
        models = await _fetch_provider_models(target_base, api_key)
    except Exception:
        return
    for m in models:
        if m["id"] == _current_model:
            _model_input_cost_per_1m = m["input_cost_per_1m"]
            _model_output_cost_per_1m = m["output_cost_per_1m"]
            return


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await _auto_resolve_model_pricing()
    yield

app.router.lifespan_context = _lifespan


@app.get("/v1/config/providers")
async def get_providers():
    """Return known providers with key status for each."""
    result = []
    for p in KNOWN_PROVIDERS:
        status = _key_status(p["url"])
        result.append({**p, "key_status": status,
                        "key_masked": _mask_key(_provider_keys.get(p["url"], ""))})
    # also include any custom URL already in use
    current_url = _config.get("upstream_base_url", "")
    known_urls = {p["url"] for p in KNOWN_PROVIDERS}
    if current_url and current_url not in known_urls:
        result.append({
            "name": "Custom", "url": current_url, "key_required": True,
            "key_status": _key_status(current_url),
            "key_masked": _mask_key(_provider_keys.get(current_url, "")),
        })
    return JSONResponse({"providers": result})


@app.get("/v1/config/key-status")
async def get_key_status(url: str = ""):
    """Return key status for a given URL."""
    target = url or _config.get("upstream_base_url", "")
    return JSONResponse({
        "url": target,
        "status": _key_status(target),
        "key_masked": _mask_key(_provider_keys.get(target, "")),
    })


@app.post("/v1/config/provider-key")
async def save_provider_key(request: Request):
    """Save an API key for a specific provider URL."""
    global UPSTREAM_KEY
    try:
        body = await request.json()
        url = body.get("url", "").strip()
        key = body.get("key", "").strip()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if not url:
        return JSONResponse({"error": "url required"}, status_code=400)
    if key:
        _provider_keys[url] = key
    elif url in _provider_keys:
        del _provider_keys[url]
    # if this is the active URL, update runtime key too
    if url == _config.get("upstream_base_url"):
        UPSTREAM_KEY = key
        _config["upstream_api_key"] = key
    try:
        _save_provider_keys()
        persisted = True
    except Exception:
        persisted = False
    return JSONResponse({"status": "saved", "persisted": persisted,
                         "key_status": _key_status(url)})


@app.get("/v1/config/settings")
async def get_settings():
    masked = dict(_config)
    current_url = masked.get("upstream_base_url", "")
    masked["upstream_api_key"] = _mask_key(masked.get("upstream_api_key", ""))
    masked["valid_modes"] = sorted(VALID_MODES)
    masked["host"] = HOST
    masked["port"] = PORT
    masked["key_status"] = _key_status(current_url)
    return JSONResponse(masked)


@app.post("/v1/config/settings")
async def update_settings(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    allowed = set(_ENV_KEY_MAP.keys())
    unknown = [k for k in body if k not in allowed]
    if unknown:
        return JSONResponse({"error": f"Unknown fields: {unknown}"}, status_code=400)

    if "context_mode" in body and body["context_mode"] not in VALID_MODES:
        return JSONResponse(
            {"error": f"Invalid context_mode. Valid: {sorted(VALID_MODES)}"},
            status_code=400,
        )

    _apply_config(body)
    try:
        _persist_env(body)
        persisted = True
    except Exception:
        persisted = False

    masked = dict(_config)
    key = masked.get("upstream_api_key", "")
    masked["upstream_api_key"] = key[:8] + "..." + key[-4:] if len(key) > 12 else ("*" * len(key))
    return JSONResponse({"status": "updated", "persisted_to_env": persisted, "settings": masked})


@app.get("/v1/requests/{request_id}/context")
async def get_request_context(request_id: str):
    ctx = _ctx_store.get(request_id)
    if ctx is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"request_id": request_id, "raw": ctx["raw"], "transformed": ctx["transformed"]}


@app.get("/v1/stats/summary")
async def stats_summary():
    return JSONResponse(_build_summary())


@app.get("/v1/stats/recent")
async def stats_recent():
    return JSONResponse(list(_recent))


@app.get("/v1/stats/stream")
async def stats_stream():
    """SSE endpoint — push live stats to the dashboard."""
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _sse_queues.append(q)

    async def event_gen() -> AsyncGenerator[str, None]:
        # send current snapshot immediately on connect
        yield f"data: {json.dumps({'stats': _build_summary(), 'latest': None})}\n\n"
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=25)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _sse_queues.remove(q)
            except ValueError:
                pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_PASSTHROUGH_BODY_KEYS = ("model", "messages", "temperature", "max_tokens", "stream", "context_mode")


def _extra_params(body: dict) -> dict:
    """Anything in the request body ShapeShifter doesn't special-case is
    forwarded to the upstream untouched: tools, tool_choice, top_p, stop,
    response_format, seed, parallel_tool_calls, stream_options, etc."""
    return {k: v for k, v in body.items() if k not in _PASSTHROUGH_BODY_KEYS}


# ---------------------------------------------------------------------------
# Feature 6 — retrieval tool: let the model ask back for collapsed content
# ---------------------------------------------------------------------------

_SHAPESHIFTER_EXPAND_TOOL = {
    "type": "function",
    "function": {
        "name": "shapeshifter_expand",
        "description": (
            "Retrieve the full, current content of a file or function shown "
            "abbreviated in this conversation as '... unchanged' or a collapsed "
            "stub. Call this if you need to see or edit something that was "
            "collapsed, instead of guessing at its content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The stub's file name, or file#function for a collapsed function.",
                },
            },
            "required": ["key"],
        },
    },
}

_MAX_RETRIEVAL_ROUNDS = 2


async def _resolve_with_retrieval(
    base_url: str, api_key: str, model: str, messages: list[dict],
    temperature: float, max_tokens: int, extra: dict, retrieval_map: dict[str, str],
) -> tuple[dict, float, int]:
    """Run the model with a synthetic `shapeshifter_expand` tool available,
    so it can ask back for anything Feature 2/3 collapsed in this request's
    context instead of guessing. Every call the model makes to that tool is
    answered instantly from `retrieval_map` — never a real network
    round-trip — and the loop continues transparently until the model
    produces a real answer or `_MAX_RETRIEVAL_ROUNDS` is reached, at which
    point the tool is withdrawn and the model is forced to answer with
    whatever it has. Returns (final_response, total_latency_ms, rounds_used).

    If the model never calls the tool, this costs exactly one upstream call
    plus the small fixed size of the tool definition — behaviorally
    identical to not having this feature at all.
    """
    tools = list(extra.get("tools") or []) + [_SHAPESHIFTER_EXPAND_TOOL]
    working_messages = list(messages)
    total_latency = 0.0

    for round_num in range(_MAX_RETRIEVAL_ROUNDS):
        resp, latency_ms = await call_upstream(
            base_url=base_url, api_key=api_key, model=model, messages=working_messages,
            temperature=temperature, max_tokens=max_tokens,
            extra_params={**extra, "tools": tools, "tool_choice": "auto"},
        )
        total_latency += latency_ms
        msg = resp["choices"][0]["message"]
        calls = [c for c in (msg.get("tool_calls") or [])
                 if c.get("function", {}).get("name") == "shapeshifter_expand"]
        if not calls:
            return resp, total_latency, round_num

        working_messages = working_messages + [msg]
        for c in calls:
            try:
                key = json.loads(c["function"].get("arguments") or "{}").get("key", "")
            except (ValueError, TypeError):
                key = ""
            content = retrieval_map.get(key, f"No collapsed content found for '{key}'.")
            working_messages.append({"role": "tool", "tool_call_id": c["id"], "content": content})

    # Cap hit — withdraw the tool entirely so the model can't ask again and
    # is forced to answer with whatever context it's already retrieved.
    resp, latency_ms = await call_upstream(
        base_url=base_url, api_key=api_key, model=model, messages=working_messages,
        temperature=temperature, max_tokens=max_tokens, extra_params=extra,
    )
    total_latency += latency_ms
    return resp, total_latency, _MAX_RETRIEVAL_ROUNDS


def _finalize_stats(
    mode: str, stats: dict, latency_ms: float, request_id: str,
    model: str, output_text: str,
    retrieval_rounds: int = 0, output_tokens_saved: int = 0,
    patches_applied: int = 0,
) -> dict:
    """Record stats/logs for a completed request (streamed or not) and return
    the `_shapeshifter` metrics block to attach to the response.

    `retrieval_rounds` (Feature 6) is reported honestly rather than hidden.
    `output_tokens_saved` is the delta between the full-file size the model
    would have had to produce and the actual patch size — 0 for requests
    where no patch was applied or patch mode is not active.
    """
    output_tokens = count_tokens(output_text)
    if LOG_RESPONSES:
        _log("responses.jsonl", {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id, "mode": mode,
            "estimated_output_tokens": output_tokens,
            "output_tokens_saved": output_tokens_saved,
            "patches_applied": patches_applied,
            "latency_ms": round(latency_ms, 1), "status": "success",
            "compression_ratio": stats["compression_ratio"],
            "reduction_pct": stats["reduction_pct"],
            "retrieval_rounds": retrieval_rounds,
        })
    _record_stats(mode, stats, latency_ms,
                  request_id=request_id, model=model,
                  output_tokens_saved=output_tokens_saved)
    result = {"request_id": request_id, "mode": mode, **stats, "latency_ms": round(latency_ms, 1)}
    if retrieval_rounds:
        result["retrieval_rounds"] = retrieval_rounds
    if output_tokens_saved:
        result["output_tokens_saved"] = output_tokens_saved
        result["patches_applied"] = patches_applied
    return result


async def _relay_stream(
    agen: AsyncGenerator[dict, None], first_chunk: dict, t0: float,
    mode: str, stats: dict, request_id: str, model: str,
    retrieval_map: dict[str, str] | None = None,
) -> AsyncGenerator[str, None]:
    """Forward upstream SSE chunks to the client as they arrive — no
    buffering of the full response — while accumulating output text so
    stats can be recorded once the stream actually ends. `_shapeshifter`
    metrics are attached to the final chunk, mirroring how they're attached
    to the full response body in the non-streaming path.

    Streaming patch note: chunks are forwarded as-is (no mid-stream
    reconstruction). After the stream ends, if the accumulated output
    contains patch markers, savings are measured and recorded in stats so
    the dashboard reflects the benefit — but the client already received the
    raw patch text, not a reconstructed file. Full Option-A reconstruction
    is only available for non-streaming requests.
    """
    output_text_parts: list[str] = []

    def _accumulate(chunk: dict) -> None:
        try:
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                output_text_parts.append(delta["content"])
        except (KeyError, IndexError, TypeError):
            pass

    pending = first_chunk
    _accumulate(pending)
    try:
        async for chunk in agen:
            yield f"data: {json.dumps(pending)}\n\n"
            pending = chunk
            _accumulate(pending)
    finally:
        latency_ms = (time.monotonic() - t0) * 1000
        output_text = "".join(output_text_parts)
        # Stats-only patch tracking for streaming: measure savings without
        # reconstructing the response (client already received the chunks).
        output_tokens_saved = patches_applied = 0
        if retrieval_map and output_text and is_patch_response(output_text):
            _, output_tokens_saved, patches_applied, _ = _process_patch_response(
                output_text, retrieval_map,
            )
        pending["_shapeshifter"] = _finalize_stats(
            mode, stats, latency_ms, request_id, model, output_text,
            output_tokens_saved=output_tokens_saved, patches_applied=patches_applied,
        )
        yield f"data: {json.dumps(pending)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    ts = datetime.utcnow().isoformat()

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
            status_code=400,
        )

    headers      = dict(request.headers)
    raw_messages = body.get("messages", [])
    model        = body.get("model") or _current_model
    temp         = float(body.get("temperature", 0.2))
    max_tok      = int(body.get("max_tokens", MAX_OUTPUT_TOKENS))
    stream_req   = bool(body.get("stream", False))
    extra        = _extra_params(body)

    # Detect agentic tool-call workflows (Cline writing files, running commands…):
    # OpenAI-style tool_calls/tool-role messages or a declared `tools` schema,
    # or Anthropic-style tool_use/tool_result content blocks. Compressing these
    # destroys the tool call chain and drops the client's own system prompt —
    # force raw passthrough of the original message structure instead.
    agentic = _is_agentic(raw_messages, body)
    if agentic:
        # Structure/roles are kept byte for byte — only an earlier tool call
        # repeated later in this same request with identical arguments may be
        # deduplicated, since only the last occurrence is needed to act on it
        # now (see _dedupe_repeated_tool_calls).
        new_messages = _dedupe_repeated_tool_calls(raw_messages)
        mode = "raw"
        raw_ctx        = "\n\n".join(_content_as_str(m) for m in raw_messages)
        transformed_ctx = "\n\n".join(_content_as_str(m) for m in new_messages)
        retrieval_map: dict[str, str] = {}  # Feature 6 is scoped to non-agentic requests only
    else:
        # Normalize multimodal text-only list content to plain strings.
        messages = [
            {**m, "content": _content_as_str(m)}
            if not isinstance(m.get("content", ""), str) else m
            for m in raw_messages
        ]
        mode = _resolve_mode(body, headers)
        if mode not in VALID_MODES:
            return JSONResponse(
                {"error": {
                    "message": f"Unknown context mode: {mode!r}",
                    "type": "invalid_request_error",
                    "code": "invalid_context_mode",
                }},
                status_code=400,
            )
        try:
            new_messages, raw_ctx, transformed_ctx, retrieval_map = _build_compressed_messages(messages, mode)
        except Exception as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "transformation_error"}},
                status_code=500,
            )

    stats = compression_stats(raw_ctx, transformed_ctx)

    # store raw + transformed context for dashboard inspection (keeps last 50)
    _ctx_store[request_id] = {"raw": raw_ctx, "transformed": transformed_ctx}
    _ctx_store_keys.append(request_id)
    if len(_ctx_store_keys) == 50 and len(_ctx_store) > 50:
        oldest = _ctx_store_keys[0]
        _ctx_store.pop(oldest, None)

    if LOG_REQUESTS:
        _log("requests.jsonl", {"timestamp": ts, "request_id": request_id,
                                "mode": mode, "auto_mode": AUTO_MODE,
                                "model": model, **stats})

    if not UPSTREAM_URL or not UPSTREAM_KEY:
        return JSONResponse(
            {"error": {"message": "UPSTREAM_BASE_URL and UPSTREAM_API_KEY must be set in .env",
                       "type": "configuration_error"}},
            status_code=500,
        )

    if stream_req:
        t0 = time.monotonic()
        agen = stream_upstream(
            base_url=UPSTREAM_URL, api_key=UPSTREAM_KEY, model=model,
            messages=new_messages, temperature=temp, max_tokens=max_tok, extra_params=extra,
        )
        try:
            first_chunk = await agen.__anext__()
        except StopAsyncIteration:
            async def _empty() -> AsyncGenerator[str, None]:
                yield "data: [DONE]\n\n"
            return StreamingResponse(_empty(), media_type="text/event-stream")
        except Exception as exc:
            return JSONResponse(
                {"error": {"message": f"Upstream error: {exc}", "type": "upstream_error"}},
                status_code=502,
            )
        return StreamingResponse(
            _relay_stream(agen, first_chunk, t0, mode, stats, request_id, model,
                          retrieval_map=retrieval_map),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    retrieval_rounds = 0
    try:
        if retrieval_map:
            # Feature 6: something in this turn's context was collapsed —
            # give the model a way to ask for it back instead of guessing.
            upstream_response, latency_ms, retrieval_rounds = await _resolve_with_retrieval(
                base_url=UPSTREAM_URL, api_key=UPSTREAM_KEY, model=model,
                messages=new_messages, temperature=temp, max_tokens=max_tok,
                extra=extra, retrieval_map=retrieval_map,
            )
        else:
            upstream_response, latency_ms = await call_upstream(
                base_url=UPSTREAM_URL, api_key=UPSTREAM_KEY, model=model,
                messages=new_messages, temperature=temp, max_tokens=max_tok, extra_params=extra,
            )
    except Exception as exc:
        return JSONResponse(
            {"error": {"message": f"Upstream error: {exc}", "type": "upstream_error"}},
            status_code=502,
        )

    output_text = ""
    try:
        output_text = upstream_response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        pass

    # Patch processing (non-streaming only — Option A: apply patches and
    # send the reconstructed full file to the client transparently).
    # Patch mode is only active for coding-session modes that have prior artifacts;
    # for all other modes retrieval_map is empty and this is a no-op.
    output_tokens_saved = patches_applied = 0
    if retrieval_map and output_text:
        output_text, output_tokens_saved, patches_applied, _ = _process_patch_response(
            output_text, retrieval_map,
        )
        if patches_applied:
            try:
                upstream_response["choices"][0]["message"]["content"] = output_text
            except (KeyError, IndexError, TypeError):
                pass

    upstream_response["_shapeshifter"] = _finalize_stats(
        mode, stats, latency_ms, request_id, model, output_text,
        retrieval_rounds, output_tokens_saved, patches_applied,
    )
    return JSONResponse(upstream_response)

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(_DASHBOARD_HTML)


_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ShapeShifter Dashboard</title>
<style>
:root {
  --bg:#0f1117; --panel:#1a1d27; --border:#2a2d3e;
  --accent:#7c6af7; --green:#22c55e; --yellow:#eab308;
  --red:#ef4444; --text:#e2e8f0; --muted:#64748b;
  --font:'JetBrains Mono','Cascadia Code',Consolas,monospace;
  --sb-w:300px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font);
     font-size:13px;display:flex;flex-direction:column;height:100vh;overflow:hidden}

/* ── Topbar ── */
#topbar{display:flex;align-items:center;gap:12px;padding:0 16px;height:44px;
        background:var(--panel);border-bottom:1px solid var(--border);flex-shrink:0;z-index:50}
#sb-btn{background:none;border:none;color:var(--muted);font-size:18px;cursor:pointer;
        line-height:1;padding:4px 6px;border-radius:4px}
#sb-btn:hover{color:var(--accent);background:rgba(124,106,247,.1)}
#topbar h1{font-size:15px;color:var(--accent);letter-spacing:1px;margin-right:auto}
.conn-dot{display:inline-block;width:8px;height:8px;border-radius:50%;
          background:var(--red);margin-right:5px;transition:background .3s}
.conn-dot.ok{background:var(--green)}
#topbar .sub{color:var(--muted);font-size:10px}
#sb-last-top{color:var(--muted);font-size:10px;max-width:320px;
             overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* ── Layout ── */
#layout{display:flex;flex:1;overflow:hidden}

/* ── Sidebar ── */
#sidebar{width:var(--sb-w);background:var(--panel);border-right:1px solid var(--border);
         display:flex;flex-direction:column;overflow-y:auto;overflow-x:hidden;
         transition:width .25s ease,opacity .25s ease;flex-shrink:0}
#sidebar.collapsed{width:0;opacity:0;pointer-events:none}
.sb-section{border-bottom:1px solid var(--border);padding:14px}
.sb-section h2{color:var(--muted);font-size:9px;text-transform:uppercase;
               letter-spacing:1px;margin-bottom:10px}

/* model row */
.model-row{display:flex;gap:6px;align-items:center}
.model-row input{flex:1;background:var(--bg);border:1px solid var(--border);
                 border-radius:4px;color:var(--text);font-family:var(--font);
                 font-size:11px;padding:5px 8px;outline:none;min-width:0}
.model-row input:focus{border-color:var(--accent)}

/* settings fields */
.sf{display:flex;flex-direction:column;gap:3px;margin-bottom:10px}
.sf label{color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:1px}
.sf input,.sf select{background:var(--bg);border:1px solid var(--border);border-radius:4px;
                     color:var(--text);font-family:var(--font);font-size:11px;
                     padding:5px 8px;outline:none;width:100%}
.sf input:focus,.sf select:focus{border-color:var(--accent)}
.sf .hint{color:var(--muted);font-size:9px}
.key-wrap{position:relative}
.key-wrap input{padding-right:46px}
.show-key{position:absolute;right:6px;top:22px;background:none;border:none;
          color:var(--muted);font-size:9px;cursor:pointer;font-family:var(--font)}
.key-badge{font-size:9px;padding:1px 6px;border-radius:3px;font-weight:bold;letter-spacing:.5px}
.key-badge.saved{background:rgba(34,197,94,.15);color:var(--green)}
.key-badge.not_set{background:rgba(239,68,68,.15);color:var(--red)}
.key-badge.not_required{background:rgba(100,116,139,.12);color:var(--muted)}
.tog-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.tog-row .hint{color:var(--muted);font-size:9px}
.tog{position:relative;display:inline-block;width:30px;height:16px;flex-shrink:0}
.tog input{opacity:0;width:0;height:0}
.tog-sl{position:absolute;inset:0;background:var(--border);border-radius:16px;
        cursor:pointer;transition:background .2s}
.tog-sl:before{content:'';position:absolute;width:10px;height:10px;
               left:3px;top:3px;background:#fff;border-radius:50%;transition:transform .2s}
.tog input:checked + .tog-sl{background:var(--accent)}
.tog input:checked + .tog-sl:before{transform:translateX(14px)}
.ro-val{background:var(--bg);border:1px solid var(--border);border-radius:4px;
        padding:5px 8px;font-size:11px;color:var(--muted)}
.sb-actions{display:flex;gap:6px;align-items:center;margin-top:4px}
#settings-status{font-size:9px;color:var(--green)}

/* ── Buttons ── */
.btn{background:var(--accent);color:#fff;border:none;border-radius:4px;
     padding:4px 12px;font-family:var(--font);font-size:10px;cursor:pointer;white-space:nowrap}
.btn:hover{opacity:.85}
.btn.sm{padding:2px 7px;font-size:9px}
.btn.ghost{background:transparent;border:1px solid var(--border);color:var(--muted)}
.btn.ghost:hover{border-color:var(--accent);color:var(--accent)}
#model-status{font-size:9px;color:var(--green);min-width:40px}

/* ── Main area ── */
#main{flex:1;overflow-y:auto;padding:16px 16px 48px}

/* ── Cards ── */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
       gap:10px;margin-bottom:20px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px}
.card.featured{border-color:var(--green);background:rgba(34,197,94,.06)}
.card-label{color:var(--muted);font-size:9px;text-transform:uppercase;
            letter-spacing:1px;margin-bottom:6px}
.card-value{font-size:26px;font-weight:bold}
.card.featured .card-value{font-size:32px}
.card-sub{color:var(--muted);font-size:9px;margin-top:3px}
.green{color:var(--green)}.yellow{color:var(--yellow)}.accent{color:var(--accent)}

/* ── Panels ── */
.panel{background:var(--panel);border:1px solid var(--border);
       border-radius:8px;padding:14px;margin-bottom:14px}
.panel h2{font-size:10px;text-transform:uppercase;letter-spacing:1px;
          color:var(--muted);margin-bottom:10px}
table{width:100%;border-collapse:collapse}
th{color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:1px;
   text-align:left;padding:6px 8px;border-bottom:1px solid var(--border)}
td{padding:6px 8px;border-bottom:1px solid var(--border);vertical-align:middle}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(124,106,247,.05)}
.mode-pill{display:inline-block;padding:2px 7px;border-radius:4px;font-size:9px;
           background:rgba(124,106,247,.15);color:var(--accent)}
.bar-wrap{width:100%;background:var(--border);border-radius:3px;height:5px}
.bar{height:5px;border-radius:3px;background:var(--green);transition:width .5s}

/* ── Feed ── */
#feed{max-height:340px;overflow-y:auto}
.feed-row{display:grid;
          grid-template-columns:58px 86px 106px 68px 68px 58px 68px 56px;
          gap:4px;padding:4px 8px;border-bottom:1px solid var(--border);
          font-size:10px;align-items:center}
.feed-row.header{color:var(--muted);font-size:9px;text-transform:uppercase;
                 letter-spacing:1px;position:sticky;top:0;background:var(--panel);z-index:1}
.feed-row.new{animation:flash .6s ease}
@keyframes flash{from{background:rgba(124,106,247,.25)}to{background:transparent}}

/* ── Models modal ── */
#models-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);
                z-index:200;align-items:center;justify-content:center}
#models-overlay.open{display:flex}
#models-modal{background:var(--panel);border:1px solid var(--border);border-radius:10px;
              width:min(820px,95vw);max-height:85vh;display:flex;flex-direction:column}
#models-header{display:flex;justify-content:space-between;align-items:center;
               padding:12px 16px;border-bottom:1px solid var(--border);flex-shrink:0;gap:10px}
#models-header h3{font-size:12px;color:var(--accent);white-space:nowrap}
#models-search{background:var(--bg);border:1px solid var(--border);border-radius:4px;
               color:var(--text);font-family:var(--font);font-size:11px;
               padding:4px 9px;outline:none;flex:1;max-width:280px}
#models-search:focus{border-color:var(--accent)}
#models-close{background:none;border:none;color:var(--muted);font-size:16px;cursor:pointer}
#models-close:hover{color:var(--text)}
#models-table-wrap{overflow-y:auto;flex:1}
#models-table{width:100%;border-collapse:collapse}
#models-table th{position:sticky;top:0;background:var(--panel);color:var(--muted);
                 font-size:9px;text-transform:uppercase;letter-spacing:1px;
                 text-align:left;padding:7px 10px;border-bottom:1px solid var(--border);
                 cursor:pointer;user-select:none}
#models-table th:hover{color:var(--text)}
#models-table td{padding:6px 10px;border-bottom:1px solid var(--border);font-size:10px;vertical-align:middle}
#models-table tr:hover td{background:rgba(108,99,255,.07);cursor:pointer}
.cost-cell{text-align:right;font-variant-numeric:tabular-nums}
.cost-free{color:var(--green)}.cost-cheap{color:var(--green)}
.cost-mid{color:var(--yellow)}.cost-expensive{color:var(--red)}
#models-status{padding:8px 16px;font-size:9px;color:var(--muted);
               border-top:1px solid var(--border);flex-shrink:0}

/* ── Context modal ── */
#modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);
               z-index:100;align-items:center;justify-content:center}
#modal-overlay.open{display:flex}
#modal{background:var(--panel);border:1px solid var(--border);border-radius:10px;
       width:min(860px,92vw);max-height:80vh;display:flex;flex-direction:column}
#modal-header{display:flex;justify-content:space-between;align-items:center;
              padding:12px 16px;border-bottom:1px solid var(--border)}
#modal-title{font-size:12px;color:var(--accent)}
#modal-meta{font-size:9px;color:var(--muted);margin-top:2px}
#modal-close{background:none;border:none;color:var(--muted);font-size:16px;cursor:pointer}
#modal-close:hover{color:var(--text)}
#modal-body{overflow-y:auto;padding:14px 16px}
#modal-ctx{white-space:pre-wrap;font-size:10px;line-height:1.6;color:var(--text);
           background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:12px}

/* ── Onboarding modal (first-run API key) ── */
#onboarding-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.82);
                    z-index:300;align-items:center;justify-content:center}
#onboarding-overlay.open{display:flex}
#onboarding-modal{background:var(--panel);border:1px solid var(--border);border-radius:10px;
                  width:min(400px,92vw);padding:20px}
#onboarding-modal h3{font-size:13px;color:var(--accent);margin-bottom:4px}
#onboarding-modal p{font-size:10px;color:var(--muted);margin-bottom:16px;line-height:1.5}
#onboarding-err{font-size:9px;color:var(--red);min-height:12px;margin-top:2px}

@media(max-width:600px){
  .cards{grid-template-columns:1fr 1fr}
  .feed-row{grid-template-columns:50px 70px 70px 55px 55px 1fr}
  .feed-row :nth-child(7),.feed-row :nth-child(8){display:none}
}
</style>
</head>
<body>

<!-- Topbar -->
<div id="topbar">
  <button id="sb-btn" onclick="toggleSidebar()" title="Toggle settings">&#9776;</button>
  <h1>&#9889; ShapeShifter</h1>
  <span id="sb-last-top"></span>
  <span class="sub"><span class="conn-dot" id="conn-dot"></span><span id="conn-label">connecting…</span></span>
</div>

<div id="layout">

<!-- ═══════════ SIDEBAR ═══════════ -->
<div id="sidebar">

  <!-- Settings -->
  <div class="sb-section" style="flex:1">
    <h2>&#9881; Settings</h2>

    <div class="sf">
      <label>Provider</label>
      <select id="s-provider" onchange="onProviderChange()">
        <option value="">— select provider —</option>
      </select>
    </div>

    <div class="sf">
      <label>Base URL</label>
      <input id="s-url" type="text" placeholder="https://openrouter.ai/api/v1" oninput="onUrlInput()" />
      <span class="hint">Any OpenAI-compatible endpoint</span>
    </div>

    <div class="sf key-wrap">
      <label style="display:flex;align-items:center;gap:6px">
        API Key
        <span id="key-badge" class="key-badge"></span>
      </label>
      <input id="s-key" type="password" placeholder="sk-…" oninput="onKeyInput()" />
      <button class="show-key" id="key-action-btn" onclick="keyActionClick()">show</button>
      <span class="hint" id="key-hint"></span>
    </div>

    <!-- Model -->
    <div class="sf">
      <label>Model</label>
      <div class="model-row">
        <input id="model-input" type="text" placeholder="e.g. deepseek/deepseek-v4-flash" />
      </div>
      <div style="display:flex;gap:6px;margin-top:6px;align-items:center">
        <button class="btn ghost" onclick="openModelsBrowser()">Browse</button>
        <button class="btn" onclick="applyModel()">Apply</button>
        <span id="model-status"></span>
      </div>
    </div>

    <div class="sf">
      <label>Context Mode</label>
      <select id="s-mode">
        <option value="hybrid">hybrid</option>
        <option value="incremental">incremental</option>
        <option value="yaml">yaml</option>
        <option value="raw">raw</option>
        <option value="minimal">minimal</option>
        <option value="json">json</option>
        <option value="table">table</option>
        <option value="symbolic">symbolic</option>
        <option value="matrix">matrix</option>
      </select>
    </div>

    <div class="sf">
      <label>Log Directory</label>
      <input id="s-logdir" type="text" placeholder="logs" />
    </div>

    <div class="tog-row">
      <label class="tog"><input type="checkbox" id="s-auto"/><span class="tog-sl"></span></label>
      <span class="hint">Auto-select mode per request</span>
    </div>
    <div class="tog-row">
      <label class="tog"><input type="checkbox" id="s-logreq"/><span class="tog-sl"></span></label>
      <span class="hint">Log requests</span>
    </div>
    <div class="tog-row">
      <label class="tog"><input type="checkbox" id="s-logres"/><span class="tog-sl"></span></label>
      <span class="hint">Log responses</span>
    </div>

    <div class="sb-actions">
      <button class="btn" onclick="saveSettings()">Save &amp; Apply</button>
      <button class="btn ghost" onclick="loadSettings()">Reset</button>
      <span id="settings-status"></span>
    </div>
  </div>

  <!-- Server info -->
  <div class="sb-section">
    <h2>&#128274; Server (read-only)</h2>
    <div class="sf"><label>Host</label><div class="ro-val" id="s-host">—</div></div>
    <div class="sf"><label>Port</label><div class="ro-val" id="s-port">—</div></div>
    <div class="sf" style="margin-bottom:0">
      <span class="hint">Restart the server to change host/port</span>
    </div>
  </div>

</div><!-- /sidebar -->

<!-- ═══════════ MAIN ═══════════ -->
<div id="main">

  <div class="cards">
    <div class="card featured">
      <div class="card-label">Est. $ Saved</div>
      <div class="card-value green" id="c-dollars">&#8212;</div>
      <div class="card-sub" id="c-dollars-sub">select model in Browse for pricing</div>
    </div>
    <div class="card">
      <div class="card-label">Tokens Saved</div>
      <div class="card-value green" id="c-saved">0</div>
      <div class="card-sub">input tokens not sent</div>
    </div>
    <div class="card">
      <div class="card-label">Output Saved</div>
      <div class="card-value green" id="c-out-saved">0</div>
      <div class="card-sub">output tokens via patches</div>
    </div>
    <div class="card">
      <div class="card-label">Avg Reduction</div>
      <div class="card-value green" id="c-reduc">&#8212;</div>
      <div class="card-sub">token reduction %</div>
    </div>
    <div class="card">
      <div class="card-label">Requests</div>
      <div class="card-value accent" id="c-reqs">0</div>
      <div class="card-sub">total this session</div>
    </div>
    <div class="card">
      <div class="card-label">Avg Compression</div>
      <div class="card-value yellow" id="c-ratio">&#8212;</div>
      <div class="card-sub">ratio vs raw</div>
    </div>
    <div class="card">
      <div class="card-label">Uptime</div>
      <div class="card-value accent" id="c-uptime">0s</div>
      <div class="card-sub">server running</div>
    </div>
  </div>

  <div class="panel">
    <h2>Mode breakdown</h2>
    <table>
      <thead><tr>
        <th>Mode</th><th>Requests</th><th>Avg In Before</th>
        <th>Avg In After</th><th>Avg Saved</th><th>Reduction %</th><th>Bar</th>
      </tr></thead>
      <tbody id="mode-tbody"></tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Live request feed</h2>
    <div id="feed">
      <div class="feed-row header">
        <span>Time</span><span>Mode</span><span>Model</span>
        <span>Tok Before</span><span>Tok After</span><span>Saved</span>
        <span>Reduc%</span><span>Context</span>
      </div>
      <div id="feed-rows"></div>
    </div>
  </div>

</div><!-- /main -->
</div><!-- /layout -->

<!-- Models browser modal -->
<div id="models-overlay" onclick="closeModelsBrowser(event)">
  <div id="models-modal">
    <div id="models-header">
      <h3>&#128269; Browse Models</h3>
      <input id="models-search" type="text" placeholder="Search model ID or name…" oninput="filterModels()" />
      <button id="models-close" onclick="closeModelsBrowser()">&#x2715;</button>
    </div>
    <div id="models-table-wrap">
      <table id="models-table">
        <thead><tr>
          <th onclick="sortModels('id')">Model ID &#8597;</th>
          <th onclick="sortModels('name')">Name &#8597;</th>
          <th onclick="sortModels('context_length')" style="text-align:right">Context &#8597;</th>
          <th onclick="sortModels('input_cost_per_1m')" style="text-align:right">Input /1M &#8597;</th>
          <th onclick="sortModels('output_cost_per_1m')" style="text-align:right">Output /1M &#8597;</th>
        </tr></thead>
        <tbody id="models-tbody"></tbody>
      </table>
    </div>
    <div id="models-status">Loading…</div>
  </div>
</div>

<!-- Context viewer modal -->
<div id="modal-overlay" onclick="closeModal(event)">
  <div id="modal">
    <div id="modal-header">
      <div>
        <div id="modal-title">Context Viewer</div>
        <div id="modal-meta"></div>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <button class="btn sm" id="btn-raw" onclick="switchView('raw')">Original</button>
        <button class="btn sm" id="btn-transformed" onclick="switchView('transformed')" style="opacity:.45">Modified</button>
        <button id="modal-close" onclick="closeModal()">&#x2715;</button>
      </div>
    </div>
    <div id="modal-body"><pre id="modal-ctx">loading…</pre></div>
  </div>
</div>

<!-- First-run onboarding modal: blocks the dashboard until a key is set for the active provider -->
<div id="onboarding-overlay">
  <div id="onboarding-modal">
    <h3>Configure a provider</h3>
    <p>Select a provider and enter its API key to start using ShapeShifter.</p>
    <div class="sf">
      <label>Provider</label>
      <select id="ob-provider" onchange="onObProviderChange()"></select>
    </div>
    <div class="sf" id="ob-key-row">
      <label>API Key</label>
      <input id="ob-key" type="password" placeholder="sk-…" onkeydown="if(event.key==='Enter')saveOnboarding()">
    </div>
    <div id="onboarding-err"></div>
    <button class="btn" style="width:100%;margin-top:6px" onclick="saveOnboarding()">Save &amp; Continue</button>
  </div>
</div>

<script>
const PORT = location.port || '8787';

// ── Sidebar ──────────────────────────────────────────────────────
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('collapsed');
}

// ── Settings + Provider management ───────────────────────────────
let _providers = [];

function onKeyInput() {
  const inp = document.getElementById('s-key');
  const btn = document.getElementById('key-action-btn');
  if (inp.value.trim()) {
    btn.textContent = 'OK';
    btn.style.color = 'var(--green)';
  } else {
    btn.textContent = inp.type === 'password' ? 'show' : 'hide';
    btn.style.color = '';
  }
}

async function keyActionClick() {
  const inp = document.getElementById('s-key');
  const btn = document.getElementById('key-action-btn');
  const key = inp.value.trim();

  if (key) {
    // Save key for current provider
    const url = currentSelectedUrl();
    if (!url) return;
    btn.textContent = '…'; btn.style.color = 'var(--yellow)';
    try {
      const r = await fetch('/v1/config/provider-key', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({url, key})
      });
      const d = await r.json();
      if (d.status === 'saved') {
        inp.value = '';
        inp.type  = 'password';
        btn.textContent = 'show'; btn.style.color = '';
        await refreshKeyBadge();
        await loadProviders();  // refresh dropdown icons
      } else {
        btn.textContent = 'err'; btn.style.color = 'var(--red)';
        setTimeout(() => { btn.textContent = 'OK'; btn.style.color = 'var(--green)'; }, 2000);
      }
    } catch(e) {
      btn.textContent = 'err'; btn.style.color = 'var(--red)';
      setTimeout(() => { btn.textContent = 'OK'; btn.style.color = 'var(--green)'; }, 2000);
    }
  } else {
    // Toggle show/hide
    inp.type = inp.type === 'password' ? 'text' : 'password';
    btn.textContent = inp.type === 'password' ? 'show' : 'hide';
    btn.style.color = '';
  }
}

let _lastKeyStatus = null;

function setKeyBadge(status, maskedKey) {
  _lastKeyStatus = status;
  const badge = document.getElementById('key-badge');
  const hint  = document.getElementById('key-hint');
  const inp   = document.getElementById('s-key');
  const btn   = document.getElementById('key-action-btn');
  badge.className = 'key-badge ' + status;
  // reset button to show/hide state whenever badge updates
  inp.type = 'password';
  btn.textContent = 'show'; btn.style.color = '';
  if (status === 'saved') {
    badge.textContent = '✓ saved';
    hint.textContent  = maskedKey || '';
    inp.placeholder   = 'type new key to replace';
  } else if (status === 'not_required') {
    badge.textContent = '○ not required';
    hint.textContent  = 'Local provider — no key needed';
    inp.placeholder   = '';
  } else {
    badge.textContent = '⚠ not provided';
    hint.textContent  = 'Type the key then click OK to save';
    inp.placeholder   = 'sk-…';
  }
}

// Returns the URL currently shown in the URL field (or the dropdown selection)
function currentSelectedUrl() {
  const urlField = document.getElementById('s-url').value.trim();
  return urlField || document.getElementById('s-provider').value || '';
}

async function refreshKeyBadge() {
  const url = currentSelectedUrl();
  if (!url) { setKeyBadge('not_set', ''); return; }
  try {
    const r = await fetch('/v1/config/key-status?url=' + encodeURIComponent(url));
    const d = await r.json();
    setKeyBadge(d.status, d.key_masked);
  } catch(e) { setKeyBadge('not_set', ''); }
}

async function onProviderChange() {
  const sel = document.getElementById('s-provider');
  const url = sel.value;
  if (!url) return;
  document.getElementById('s-url').value = url;
  document.getElementById('s-key').value = '';  // always clear — key belongs to that provider
  await refreshKeyBadge();
}

async function onUrlInput() {
  const url = document.getElementById('s-url').value.trim();
  if (!url) return;
  // sync provider dropdown selection to matching entry (or blank for custom)
  const sel = document.getElementById('s-provider');
  const match = Array.from(sel.options).find(o => o.value === url);
  sel.value = match ? url : '';
  // clear key field — switching URL means we're targeting a different provider
  document.getElementById('s-key').value = '';
  await refreshKeyBadge();
}

async function loadProviders() {
  const r = await fetch('/v1/config/providers');
  const d = await r.json();
  _providers = d.providers || [];
  const sel = document.getElementById('s-provider');
  const currentUrl = document.getElementById('s-url').value;
  // rebuild options (keep placeholder)
  while (sel.options.length > 1) sel.remove(1);
  _providers.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.url;
    const icon = p.key_status === 'saved' ? ' ✓' : p.key_status === 'not_required' ? '' : ' ⚠';
    opt.textContent = p.name + icon;
    sel.appendChild(opt);
  });
  if (currentUrl) sel.value = currentUrl;
}

async function loadSettings() {
  const r = await fetch('/v1/config/settings');
  const d = await r.json();
  const url = d.upstream_base_url || '';
  document.getElementById('s-url').value   = url;
  document.getElementById('s-key').value   = '';
  document.getElementById('s-mode').value  = d.context_mode || 'hybrid';
  document.getElementById('s-logdir').value= d.log_dir || 'logs';
  document.getElementById('s-auto').checked    = !!d.auto_mode;
  document.getElementById('s-logreq').checked  = !!d.log_requests;
  document.getElementById('s-logres').checked  = !!d.log_responses;
  document.getElementById('s-host').textContent= d.host || '—';
  document.getElementById('s-port').textContent= d.port || '—';
  document.getElementById('model-input').value = d.default_model || '';
  // load providers first so the dropdown is populated
  await loadProviders();
  // badge reflects the KEY STATUS of the currently selected URL, not _config's key
  await refreshKeyBadge();
  maybeShowOnboarding();
}

// ── First-run onboarding (blocks dashboard until a key is set) ────
function maybeShowOnboarding() {
  const overlay = document.getElementById('onboarding-overlay');
  if (_lastKeyStatus === 'not_set') {
    populateOnboarding();
    overlay.classList.add('open');
  } else {
    overlay.classList.remove('open');
  }
}

function populateOnboarding() {
  const sel = document.getElementById('ob-provider');
  sel.innerHTML = '';
  _providers.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.url;
    const icon = p.key_status === 'saved' ? ' ✓' : p.key_status === 'not_required' ? '' : ' ⚠';
    opt.textContent = p.name + icon;
    sel.appendChild(opt);
  });
  const current = currentSelectedUrl();
  if (current && Array.from(sel.options).some(o => o.value === current)) sel.value = current;
  onObProviderChange();
}

function onObProviderChange() {
  const sel = document.getElementById('ob-provider');
  const p = _providers.find(x => x.url === sel.value);
  document.getElementById('ob-key-row').style.display = (p && p.key_status === 'not_required') ? 'none' : '';
  document.getElementById('ob-key').value = '';
  document.getElementById('onboarding-err').textContent = '';
}

async function saveOnboarding() {
  const sel = document.getElementById('ob-provider');
  const url = sel.value;
  const err = document.getElementById('onboarding-err');
  if (!url) { err.textContent = 'Select a provider.'; return; }
  const p = _providers.find(x => x.url === url);
  const needsKey = !p || p.key_status !== 'not_required';
  const key = document.getElementById('ob-key').value.trim();
  if (needsKey && !key) { err.textContent = 'API key required for this provider.'; return; }
  err.textContent = '';
  try {
    if (needsKey) {
      const r = await fetch('/v1/config/provider-key', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({url, key})
      });
      const d = await r.json();
      if (d.status !== 'saved') { err.textContent = 'Failed to save key.'; return; }
    }
    // switch the active provider to the one chosen here
    await fetch('/v1/config/settings', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({upstream_base_url: url})
    });
    await loadSettings();
  } catch(e) { err.textContent = 'Network error.'; }
}

async function saveSettings() {
  const st  = document.getElementById('settings-status');
  const url = document.getElementById('s-url').value.trim();
  const key = document.getElementById('s-key').value.trim();
  st.style.color = 'var(--yellow)'; st.textContent = 'saving…';

  // if a key was entered, save it to provider-key store first
  if (key && url) {
    await fetch('/v1/config/provider-key', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({url, key})
    });
  }

  const payload = {
    upstream_base_url: url,
    context_mode:      document.getElementById('s-mode').value,
    log_dir:           document.getElementById('s-logdir').value.trim(),
    auto_mode:         document.getElementById('s-auto').checked,
    log_requests:      document.getElementById('s-logreq').checked,
    log_responses:     document.getElementById('s-logres').checked,
  };
  if (key) payload.upstream_api_key = key;

  try {
    const r = await fetch('/v1/config/settings', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    if (d.status === 'updated') {
      st.style.color = 'var(--green)';
      st.textContent = d.persisted_to_env ? 'Saved ✓' : 'Applied';
      await loadSettings();
    } else { st.style.color='var(--red)'; st.textContent = d.error||'error'; }
  } catch(e) { st.style.color='var(--red)'; st.textContent='network error'; }
  setTimeout(()=>{ st.textContent=''; }, 3000);
}

// ── Model quick bar ───────────────────────────────────────────────
async function applyModel() {
  const val = document.getElementById('model-input').value.trim();
  if (!val) return;
  const st = document.getElementById('model-status');
  st.style.color='var(--yellow)'; st.textContent='…';
  try {
    const r = await fetch('/v1/config/model',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({model:val})});
    const d = await r.json();
    st.style.color = d.status==='updated' ? 'var(--green)' : 'var(--red)';
    st.textContent  = d.status==='updated' ? 'ok' : d.error||'err';
  } catch(e){ st.style.color='var(--red)'; st.textContent='err'; }
  setTimeout(()=>{ st.textContent=''; }, 2500);
}
document.getElementById('model-input').addEventListener('keydown', e=>{
  if(e.key==='Enter') applyModel();
});

// ── Uptime ────────────────────────────────────────────────────────
let _uptime = 0;
function fmtUptime(s) {
  if(s<60) return s+'s';
  if(s<3600) return Math.floor(s/60)+'m '+(s%60)+'s';
  return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m';
}
setInterval(()=>{ _uptime++; document.getElementById('c-uptime').textContent=fmtUptime(_uptime); },1000);

// ── Cards + mode table ────────────────────────────────────────────
function setConn(ok) {
  const dot=document.getElementById('conn-dot'), lbl=document.getElementById('conn-label');
  dot.className='conn-dot'+(ok?' ok':''); lbl.textContent=ok?'live':'reconnecting…';
}
function updateCards(s) {
  document.getElementById('c-reqs').textContent     = s.total_requests;
  document.getElementById('c-saved').textContent    = s.total_tokens_saved.toLocaleString();
  document.getElementById('c-out-saved').textContent = (s.total_output_tokens_saved || 0).toLocaleString();
  document.getElementById('c-ratio').textContent    = s.total_requests ? s.avg_ratio.toFixed(3) : '\\u2014';
  document.getElementById('c-reduc').textContent    = s.total_requests ? s.avg_reduction_pct.toFixed(1)+'%' : '\\u2014';
  _uptime = s.uptime_s;
  // dollars saved card
  const dEl  = document.getElementById('c-dollars');
  const dSub = document.getElementById('c-dollars-sub');
  if (s.dollars_saved !== null && s.dollars_saved !== undefined) {
    const v = s.dollars_saved;
    dEl.textContent  = v < 0.001 ? '$' + v.toFixed(6) : v < 0.01 ? '$' + v.toFixed(4) : '$' + v.toFixed(4);
    const cost = s.model_input_cost_per_1m;
    dSub.textContent = cost !== null && cost !== undefined
      ? `@ $${cost.toFixed(4)}/1M input tok`
      : 'estimated savings';
  } else {
    dEl.textContent  = '\\u2014';
    dSub.textContent = 'select model in Browse for pricing';
  }
}
function updateModeTable(byMode) {
  const tbody = document.getElementById('mode-tbody');
  tbody.innerHTML = '';
  const modes = Object.entries(byMode).sort((a,b)=>b[1].count-a[1].count);
  modes.forEach(([mode,v])=>{
    const pct=Math.max(0,v.avg_reduction_pct);
    const col=pct>=50?'#22c55e':pct>=20?'#eab308':'#ef4444';
    const tr=document.createElement('tr');
    tr.innerHTML=`
      <td><span class="mode-pill">${mode}</span></td>
      <td>${v.count}</td><td>${v.avg_before.toLocaleString()}</td>
      <td>${v.avg_after.toLocaleString()}</td>
      <td class="green">${v.avg_saved.toLocaleString()}</td>
      <td style="color:${col}">${v.avg_reduction_pct>0?'-':'+'}${Math.abs(v.avg_reduction_pct)}%</td>
      <td><div class="bar-wrap"><div class="bar" style="width:${Math.min(100,pct)}%;background:${col}"></div></div></td>`;
    tbody.appendChild(tr);
  });
  if(!modes.length) tbody.innerHTML='<tr><td colspan="7" style="color:var(--muted);padding:12px 8px">No requests yet</td></tr>';
}

// ── Feed ──────────────────────────────────────────────────────────
function addFeedRow(e) {
  const container=document.getElementById('feed-rows');
  const pctColor=e.reduction_pct>=50?'#22c55e':e.reduction_pct>=20?'#eab308':'#ef4444';
  const modelShort=(e.model||'').split('/').pop()||'—';
  const reqId=e.request_id||'', mode=e.mode||'—', model=e.model||'—';
  const row=document.createElement('div');
  row.className='feed-row new';
  row.innerHTML=`
    <span style="color:var(--muted)">${e.ts}</span>
    <span><span class="mode-pill">${mode}</span></span>
    <span style="color:var(--muted);font-size:9px" title="${model}">${modelShort}</span>
    <span>${e.tok_before.toLocaleString()}</span>
    <span>${e.tok_after.toLocaleString()}</span>
    <span class="green">${e.tok_saved.toLocaleString()}</span>
    <span style="color:${pctColor}">${e.reduction_pct>0?'-':'+'}${Math.abs(e.reduction_pct)}%</span>
    <span>${reqId?`<button class="btn sm" onclick="openCtx('${reqId}','${mode}','${model.replace(/'/g,"\\\\'")}')">view</button>`:'—'}</span>`;
  container.insertBefore(row,container.firstChild);
  while(container.children.length>50) container.removeChild(container.lastChild);
  document.getElementById('sb-last-top').textContent=
    `${mode} (${modelShort}) — saved ${e.tok_saved.toLocaleString()} tok`;
}

// ── Models browser ────────────────────────────────────────────────
let _allModels=[],_sortKey='id',_sortAsc=true;
function costClass(v){if(v===null||v===undefined)return '';if(v===0)return 'cost-free';if(v<0.5)return 'cost-cheap';if(v<5)return 'cost-mid';return 'cost-expensive';}
function fmtCost(v){if(v===null||v===undefined)return '<span style="color:var(--muted)">—</span>';if(v===0)return '<span class="cost-free">free</span>';return `<span class="${costClass(v)}">$${v.toFixed(4)}</span>`;}
function fmtCtx(v){if(!v)return '<span style="color:var(--muted)">—</span>';return v>=1000?(v/1000).toFixed(0)+'K':v;}
function renderModels(){
  const q=(document.getElementById('models-search').value||'').toLowerCase();
  const tbody=document.getElementById('models-tbody');
  let rows=_allModels.filter(m=>!q||m.id.toLowerCase().includes(q)||(m.name||'').toLowerCase().includes(q));
  rows.sort((a,b)=>{
    let av=a[_sortKey],bv=b[_sortKey];
    if(av===null||av===undefined)av=_sortAsc?Infinity:-Infinity;
    if(bv===null||bv===undefined)bv=_sortAsc?Infinity:-Infinity;
    if(typeof av==='string')return _sortAsc?av.localeCompare(bv):bv.localeCompare(av);
    return _sortAsc?av-bv:bv-av;
  });
  tbody.innerHTML=rows.map(m=>{
    const cost = m.input_cost_per_1m !== null && m.input_cost_per_1m !== undefined ? m.input_cost_per_1m : 'null';
    const outCost = m.output_cost_per_1m !== null && m.output_cost_per_1m !== undefined ? m.output_cost_per_1m : 'null';
    return `<tr onclick="selectModel('${m.id.replace(/'/g,"\\\\'")}', ${cost}, ${outCost})">
      <td style="color:var(--accent);font-size:10px">${m.id}</td>
      <td style="color:var(--muted);font-size:9px">${m.name!==m.id?m.name:''}</td>
      <td class="cost-cell" style="color:var(--muted)">${fmtCtx(m.context_length)}</td>
      <td class="cost-cell">${fmtCost(m.input_cost_per_1m)}</td>
      <td class="cost-cell">${fmtCost(m.output_cost_per_1m)}</td>
    </tr>`;
  }).join('');
  document.getElementById('models-status').textContent=`${rows.length} model${rows.length!==1?'s':''} — click to select`;
}
function filterModels(){renderModels();}
function sortModels(key){if(_sortKey===key)_sortAsc=!_sortAsc;else{_sortKey=key;_sortAsc=true;}renderModels();}
function selectModel(id, inputCostPer1m, outputCostPer1m) {
  document.getElementById('model-input').value = id;
  closeModelsBrowser();
  // pass pricing to server so dollar savings can be calculated
  const body = {model: id};
  if (inputCostPer1m !== null && inputCostPer1m !== undefined) body.input_cost_per_1m = inputCostPer1m;
  if (outputCostPer1m !== null && outputCostPer1m !== undefined) body.output_cost_per_1m = outputCostPer1m;
  const st = document.getElementById('model-status');
  st.style.color = 'var(--yellow)'; st.textContent = '…';
  fetch('/v1/config/model', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})
    .then(r => r.json())
    .then(d => {
      st.style.color = d.status === 'updated' ? 'var(--green)' : 'var(--red)';
      st.textContent  = d.status === 'updated' ? 'ok' : d.error || 'err';
      setTimeout(() => { st.textContent = ''; }, 2500);
    })
    .catch(() => { st.style.color = 'var(--red)'; st.textContent = 'err'; });
}
async function openModelsBrowser(){
  const providerUrl = currentSelectedUrl();
  document.getElementById('models-overlay').classList.add('open');
  document.getElementById('models-tbody').innerHTML='';
  document.getElementById('models-search').value='';
  const providerName = (() => {
    const sel = document.getElementById('s-provider');
    const match = Array.from(sel.options).find(o => o.value === providerUrl);
    return match ? match.textContent.replace(/[✓⚠]/g,'').trim() : (providerUrl || 'upstream');
  })();
  document.getElementById('models-status').textContent = `Fetching models from ${providerName}…`;
  try{
    const qs = providerUrl ? '?url=' + encodeURIComponent(providerUrl) : '';
    const r = await fetch('/v1/upstream/models' + qs);
    const d = await r.json();
    if(d.error){document.getElementById('models-status').textContent='Error: '+d.error;return;}
    _allModels=d.data||[];
    renderModels();
  }catch(e){document.getElementById('models-status').textContent='Network error: '+e.message;}
}
function closeModelsBrowser(e){if(e&&e.target!==document.getElementById('models-overlay'))return;document.getElementById('models-overlay').classList.remove('open');}

// ── Context viewer ────────────────────────────────────────────────
let _ctxData={raw:'',transformed:''},_ctxView='raw';
function switchView(which){
  _ctxView=which;
  document.getElementById('modal-ctx').textContent=_ctxData[which]||'(empty)';
  document.getElementById('btn-raw').style.opacity=which==='raw'?'1':'.45';
  document.getElementById('btn-transformed').style.opacity=which==='transformed'?'1':'.45';
  document.getElementById('modal-title').textContent=which==='raw'?'Original Context':'Modified Context (wrapper output)';
}
async function openCtx(reqId,mode,model){
  const overlay=document.getElementById('modal-overlay');
  document.getElementById('modal-meta').textContent=`${reqId} · mode: ${mode} · model: ${model}`;
  document.getElementById('modal-ctx').textContent='loading…';
  _ctxData={raw:'',transformed:''};
  overlay.classList.add('open'); switchView('raw');
  try{
    const r=await fetch(`/v1/requests/${reqId}/context`);
    const d=await r.json();
    _ctxData.raw=d.raw||'(empty)'; _ctxData.transformed=d.transformed||'(empty)';
    document.getElementById('modal-ctx').textContent=_ctxData[_ctxView];
  }catch(e){document.getElementById('modal-ctx').textContent='Error loading context.';}
}
function closeModal(e){if(e&&e.target!==document.getElementById('modal-overlay'))return;document.getElementById('modal-overlay').classList.remove('open');}

document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){
    document.getElementById('models-overlay').classList.remove('open');
    document.getElementById('modal-overlay').classList.remove('open');
  }
});

// ── SSE ───────────────────────────────────────────────────────────
function connect(){
  const es=new EventSource('/v1/stats/stream');
  es.onopen=()=>setConn(true);
  es.onerror=()=>{setConn(false);es.close();setTimeout(connect,3000);};
  es.onmessage=(ev)=>{
    try{
      const msg=JSON.parse(ev.data);
      updateCards(msg.stats);
      updateModeTable(msg.stats.by_mode);
      if(msg.latest)addFeedRow(msg.latest);
    }catch(e){}
  };
}

loadSettings();
connect();
updateModeTable({});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n  ShapeShifter v0.2  —  http://{HOST}:{PORT}/v1")
    print(f"  Dashboard        —  http://{HOST}:{PORT}/dashboard")
    print(f"  Alfa1            —  http://{HOST}:{PORT}/alfa1")
    print(f"  Mode: {CONTEXT_MODE} | Auto: {AUTO_MODE} | Upstream: {UPSTREAM_URL or '(not set)'}\n")
    uvicorn.run("wrapper_server:app", host=HOST, port=PORT, reload=False)
