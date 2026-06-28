"""ShapeShifter — local OpenAI-compatible context compression proxy."""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from llm_client import call_upstream
from mode_selector import choose_mode
from output_contracts import build_system_prompt, detect_contract_type
from token_counter import compression_stats, count_tokens
from transformers import VALID_MODES, apply_transform

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOST              = os.getenv("WRAPPER_HOST", "127.0.0.1")
PORT              = int(os.getenv("WRAPPER_PORT", "8787"))
UPSTREAM_URL      = os.getenv("UPSTREAM_BASE_URL", "")
UPSTREAM_KEY      = os.getenv("UPSTREAM_API_KEY", "")
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
    "total_requests":    0,
    "total_tokens_before": 0,
    "total_tokens_after":  0,
    "total_tokens_saved":  0,
    "by_mode": {m: {"count": 0, "tok_before": 0, "tok_after": 0, "tok_saved": 0}
                for m in VALID_MODES},
}
_recent: deque[dict] = deque(maxlen=50)
_ctx_store: dict[str, dict] = {}   # request_id -> {raw, transformed} (last 50)
_ctx_store_keys: deque[str] = deque(maxlen=50)
_sse_queues: list[asyncio.Queue] = []
_current_model: str = DEFAULT_MODEL


def _record_stats(mode: str, s: dict, latency_ms: float, request_id: str = "", model: str = "") -> None:
    _stats["total_requests"]      += 1
    _stats["total_tokens_before"] += s["tokens_before"]
    _stats["total_tokens_after"]  += s["tokens_after"]
    _stats["total_tokens_saved"]  += s["tokens_saved"]
    bm = _stats["by_mode"].setdefault(mode, {"count": 0, "tok_before": 0, "tok_after": 0, "tok_saved": 0})
    bm["count"]      += 1
    bm["tok_before"] += s["tokens_before"]
    bm["tok_after"]  += s["tokens_after"]
    bm["tok_saved"]  += s["tokens_saved"]

    entry = {
        "ts":            datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "request_id":    request_id,
        "mode":          mode,
        "model":         model,
        "tok_before":    s["tokens_before"],
        "tok_after":     s["tokens_after"],
        "tok_saved":     s["tokens_saved"],
        "reduction_pct": s["reduction_pct"],
        "latency_ms":    round(latency_ms, 0),
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
    return {
        "total_requests":    _stats["total_requests"],
        "total_tokens_saved": _stats["total_tokens_saved"],
        "total_tokens_before": _stats["total_tokens_before"],
        "avg_ratio":         round(avg_ratio, 3),
        "avg_reduction_pct": round((1 - avg_ratio) * 100, 1),
        "uptime_s":          uptime_s,
        "by_mode":           by_mode_out,
    }

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ShapeShifter", version="0.2.0")


def _log(filename: str, record: dict) -> None:
    path = LOG_DIR / filename
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
        raw_ctx  = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        user_req = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return choose_mode(raw_ctx, user_req)
    return "hybrid"


def _build_compressed_messages(
    original_messages: list[dict], mode: str
) -> tuple[list[dict], str, str]:
    # Separate history (all but last user message) from the current user request.
    # Only history is compressed — the current user message is always sent intact
    # so pasted code, file contents, or inline examples are never stripped.
    last_user_idx = next(
        (i for i in range(len(original_messages) - 1, -1, -1)
         if original_messages[i].get("role") == "user"),
        None,
    )
    if last_user_idx is not None and last_user_idx > 0:
        history   = original_messages[:last_user_idx]
        current   = original_messages[last_user_idx]
    else:
        history   = []
        current   = original_messages[-1] if original_messages else {"role": "user", "content": ""}

    raw_ctx, transformed_ctx = apply_transform(mode, history) if history else ("", "")
    contract_type  = detect_contract_type(original_messages)
    system_content = build_system_prompt(mode, contract_type)

    new_messages: list[dict] = [{"role": "system", "content": system_content}]
    if transformed_ctx:
        new_messages.append({"role": "user",      "content": transformed_ctx})
        new_messages.append({"role": "assistant",  "content": "Understood."})
    new_messages.append(current)

    # stats are computed over the full raw context vs compressed history
    full_raw = apply_transform("raw", original_messages)[0] if history else current.get("content", "")
    return new_messages, full_raw, transformed_ctx

# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0", "uptime_s": int(time.monotonic() - _start_time)}


@app.get("/v1/config/model")
async def get_model():
    return {"model": _current_model}


@app.post("/v1/config/model")
async def set_model(request: Request):
    global _current_model
    try:
        body = await request.json()
        new_model = body.get("model", "").strip()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if not new_model:
        return JSONResponse({"error": "model field required"}, status_code=400)
    _current_model = new_model
    return {"model": _current_model, "status": "updated"}


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


def _to_sse_stream(upstream_response: dict, request_id: str) -> "AsyncGenerator[str, None]":
    """Convert a non-streaming upstream response to OpenAI SSE format."""
    async def _gen() -> AsyncGenerator[str, None]:
        chunk_id = upstream_response.get("id", f"chatcmpl-{request_id}")
        model    = upstream_response.get("model", DEFAULT_MODEL)
        content  = ""
        try:
            content = upstream_response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            pass

        # role delta
        role_chunk = {
            "id": chunk_id, "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        # content delta (single chunk — simpler, Cline handles it fine)
        if content:
            content_chunk = {
                "id": chunk_id, "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"

        # finish chunk
        finish_chunk = {
            "id": chunk_id, "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(finish_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return _gen()


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

    headers    = dict(request.headers)
    messages   = body.get("messages", [])
    model      = body.get("model") or _current_model
    temp       = float(body.get("temperature", 0.2))
    max_tok    = int(body.get("max_tokens", MAX_OUTPUT_TOKENS))
    stream_req = bool(body.get("stream", False))
    mode       = _resolve_mode(body, headers)

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
        new_messages, raw_ctx, transformed_ctx = _build_compressed_messages(messages, mode)
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

    latency_ms = 0.0
    try:
        upstream_response, latency_ms = await call_upstream(
            base_url=UPSTREAM_URL, api_key=UPSTREAM_KEY, model=model,
            messages=new_messages, temperature=temp, max_tokens=max_tok,
        )
    except Exception as exc:
        return JSONResponse(
            {"error": {"message": f"Upstream error: {exc}", "type": "upstream_error"}},
            status_code=502,
        )

    output_text = ""
    try:
        output_text = upstream_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        pass
    output_tokens = count_tokens(output_text)

    if LOG_RESPONSES:
        _log("responses.jsonl", {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id, "mode": mode,
            "estimated_output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1), "status": "success",
            "compression_ratio": stats["compression_ratio"],
            "reduction_pct": stats["reduction_pct"],
        })

    _record_stats(mode, stats, latency_ms, request_id=request_id, model=model)

    upstream_response["_shapeshifter"] = {
        "request_id": request_id, "mode": mode,
        **stats, "latency_ms": round(latency_ms, 1),
    }

    if stream_req:
        return StreamingResponse(
            _to_sse_stream(upstream_response, request_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ShapeShifter Dashboard</title>
<style>
  :root {
    --bg: #0f1117; --panel: #1a1d27; --border: #2a2d3e;
    --accent: #7c6af7; --green: #22c55e; --yellow: #eab308;
    --red: #ef4444; --text: #e2e8f0; --muted: #64748b;
    --font: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font);
         font-size: 13px; min-height: 100vh; padding: 20px 20px 48px; }
  h1 { font-size: 18px; color: var(--accent); margin-bottom: 6px; letter-spacing: 1px; }
  .subtitle { color: var(--muted); font-size: 11px; margin-bottom: 16px; }

  /* model bar */
  .model-bar { display: flex; align-items: center; gap: 10px; margin-bottom: 20px;
               background: var(--panel); border: 1px solid var(--border);
               border-radius: 8px; padding: 10px 14px; }
  .model-bar label { color: var(--muted); font-size: 10px; text-transform: uppercase;
                     letter-spacing: 1px; white-space: nowrap; }
  .model-bar input { flex: 1; background: var(--bg); border: 1px solid var(--border);
                     border-radius: 4px; color: var(--text); font-family: var(--font);
                     font-size: 12px; padding: 5px 9px; outline: none; }
  .model-bar input:focus { border-color: var(--accent); }
  .btn { background: var(--accent); color: #fff; border: none; border-radius: 4px;
         padding: 5px 14px; font-family: var(--font); font-size: 11px;
         cursor: pointer; white-space: nowrap; }
  .btn:hover { opacity: 0.85; }
  .btn.sm { padding: 2px 9px; font-size: 10px; }
  #model-status { font-size: 10px; color: var(--green); min-width: 60px; }

  /* top cards */
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
           gap: 12px; margin-bottom: 24px; }
  .card { background: var(--panel); border: 1px solid var(--border);
          border-radius: 8px; padding: 16px; }
  .card-label { color: var(--muted); font-size: 10px; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 8px; }
  .card-value { font-size: 28px; font-weight: bold; }
  .card-sub   { color: var(--muted); font-size: 10px; margin-top: 4px; }
  .green { color: var(--green); }
  .yellow { color: var(--yellow); }
  .accent { color: var(--accent); }

  /* tables */
  .panel { background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px;
              color: var(--muted); margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; }
  th { color: var(--muted); font-size: 10px; text-transform: uppercase;
       letter-spacing: 1px; text-align: left; padding: 6px 10px;
       border-bottom: 1px solid var(--border); }
  td { padding: 7px 10px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(124,106,247,0.06); }
  .mode-pill { display: inline-block; padding: 2px 8px; border-radius: 4px;
               font-size: 10px; background: rgba(124,106,247,0.15);
               color: var(--accent); }
  .bar-wrap { width: 100%; background: var(--border); border-radius: 3px; height: 6px; }
  .bar { height: 6px; border-radius: 3px; background: var(--green);
         transition: width 0.5s ease; }

  /* live feed */
  #feed { max-height: 380px; overflow-y: auto; }
  .feed-row { display: grid;
              grid-template-columns: 62px 90px 110px 72px 72px 62px 72px 60px;
              gap: 6px; padding: 5px 10px; border-bottom: 1px solid var(--border);
              font-size: 11px; align-items: center; }
  .feed-row.header { color: var(--muted); font-size: 10px; text-transform: uppercase;
                     letter-spacing: 1px; position: sticky; top: 0;
                     background: var(--panel); z-index: 1; }
  .feed-row.new { animation: flash 0.6s ease; }
  @keyframes flash { from { background: rgba(124,106,247,0.25); } to { background: transparent; } }

  /* modal */
  #modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7);
                   z-index: 100; align-items: center; justify-content: center; }
  #modal-overlay.open { display: flex; }
  #modal { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
           width: min(860px, 92vw); max-height: 80vh; display: flex; flex-direction: column; }
  #modal-header { display: flex; justify-content: space-between; align-items: center;
                  padding: 14px 18px; border-bottom: 1px solid var(--border); }
  #modal-title { font-size: 13px; color: var(--accent); }
  #modal-meta  { font-size: 10px; color: var(--muted); margin-top: 3px; }
  #modal-close { background: none; border: none; color: var(--muted); font-size: 18px;
                 cursor: pointer; line-height: 1; }
  #modal-close:hover { color: var(--text); }
  #modal-body { overflow-y: auto; padding: 16px 18px; }
  #modal-ctx { white-space: pre-wrap; font-size: 11px; line-height: 1.6;
               color: var(--text); background: var(--bg); border: 1px solid var(--border);
               border-radius: 6px; padding: 14px; }

  /* status bar */
  #statusbar { position: fixed; bottom: 0; left: 0; right: 0;
               background: var(--panel); border-top: 1px solid var(--border);
               padding: 6px 20px; font-size: 10px; color: var(--muted);
               display: flex; gap: 20px; }
  #conn-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
              background: var(--red); margin-right: 6px; transition: background 0.3s; }
  #conn-dot.ok { background: var(--green); }
  @media (max-width: 700px) {
    .cards { grid-template-columns: 1fr 1fr; }
    .feed-row { grid-template-columns: 55px 80px 80px 60px 60px 1fr; }
    .feed-row :nth-child(7), .feed-row :nth-child(8) { display: none; }
  }
</style>
</head>
<body>
<h1>&#9889; ShapeShifter</h1>
<p class="subtitle">Real-time token compression dashboard &nbsp;&middot;&nbsp;
  <span id="conn-dot"></span><span id="conn-label">connecting...</span>
</p>

<!-- Model selector -->
<div class="model-bar">
  <label>Model</label>
  <input id="model-input" type="text" placeholder="e.g. deepseek/deepseek-v4-flash" />
  <button class="btn" onclick="applyModel()">Apply</button>
  <span id="model-status"></span>
</div>

<div class="cards">
  <div class="card">
    <div class="card-label">Requests</div>
    <div class="card-value accent" id="c-reqs">0</div>
    <div class="card-sub">total this session</div>
  </div>
  <div class="card">
    <div class="card-label">Tokens Saved</div>
    <div class="card-value green" id="c-saved">0</div>
    <div class="card-sub">input tokens not sent to cloud</div>
  </div>
  <div class="card">
    <div class="card-label">Avg Compression</div>
    <div class="card-value yellow" id="c-ratio">&#8212;</div>
    <div class="card-sub">ratio vs raw context</div>
  </div>
  <div class="card">
    <div class="card-label">Avg Reduction</div>
    <div class="card-value green" id="c-reduc">&#8212;</div>
    <div class="card-sub">token reduction %</div>
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
    <thead>
      <tr>
        <th>Mode</th><th>Requests</th><th>Avg In Before</th>
        <th>Avg In After</th><th>Avg Saved</th><th>Reduction %</th><th>Bar</th>
      </tr>
    </thead>
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

<!-- Context modal -->
<div id="modal-overlay" onclick="closeModal(event)">
  <div id="modal">
    <div id="modal-header">
      <div>
        <div id="modal-title">Context Viewer</div>
        <div id="modal-meta"></div>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <button class="btn sm" id="btn-raw"         onclick="switchView('raw')">Original</button>
        <button class="btn sm" id="btn-transformed" onclick="switchView('transformed')" style="opacity:0.45">Modified</button>
        <button id="modal-close" onclick="closeModal()">&#x2715;</button>
      </div>
    </div>
    <div id="modal-body">
      <pre id="modal-ctx">loading...</pre>
    </div>
  </div>
</div>

<div id="statusbar">
  <span><span id="conn-dot2"></span> SSE stream</span>
  <span id="sb-last">waiting for requests...</span>
  <span>endpoint: <strong>http://127.0.0.1:__PORT__/v1/chat/completions</strong></span>
</div>

<script>
const PORT = location.port || '8787';
document.querySelectorAll('strong').forEach(el => {
  el.textContent = el.textContent.replace('__PORT__', PORT);
});

// load current model on start
fetch('/v1/config/model').then(r => r.json()).then(d => {
  document.getElementById('model-input').value = d.model || '';
});

async function applyModel() {
  const val = document.getElementById('model-input').value.trim();
  if (!val) return;
  const st = document.getElementById('model-status');
  st.style.color = 'var(--yellow)';
  st.textContent = 'saving...';
  try {
    const r = await fetch('/v1/config/model', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: val})
    });
    const d = await r.json();
    if (d.status === 'updated') {
      st.style.color = 'var(--green)';
      st.textContent = 'saved!';
    } else {
      st.style.color = 'var(--red)';
      st.textContent = d.error || 'error';
    }
  } catch(e) {
    st.style.color = 'var(--red)';
    st.textContent = 'error';
  }
  setTimeout(() => { st.textContent = ''; }, 2500);
}

document.getElementById('model-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') applyModel();
});

// uptime counter
let _uptime = 0;
function fmtUptime(s) {
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s/60) + 'm ' + (s%60) + 's';
  return Math.floor(s/3600) + 'h ' + Math.floor((s%3600)/60) + 'm';
}
setInterval(() => {
  _uptime++;
  document.getElementById('c-uptime').textContent = fmtUptime(_uptime);
}, 1000);

function setConn(ok) {
  const dot  = document.getElementById('conn-dot');
  const dot2 = document.getElementById('conn-dot2');
  const lbl  = document.getElementById('conn-label');
  [dot, dot2].forEach(d => { if (d) d.className = ok ? 'ok' : ''; });
  lbl.textContent = ok ? 'live' : 'reconnecting...';
}

function updateCards(s) {
  document.getElementById('c-reqs').textContent  = s.total_requests;
  document.getElementById('c-saved').textContent = s.total_tokens_saved.toLocaleString();
  document.getElementById('c-ratio').textContent = s.total_requests ? s.avg_ratio.toFixed(3) : '\\u2014';
  document.getElementById('c-reduc').textContent = s.total_requests ? s.avg_reduction_pct.toFixed(1) + '%' : '\\u2014';
  _uptime = s.uptime_s;
  document.getElementById('c-uptime').textContent = fmtUptime(s.uptime_s);
}

function updateModeTable(byMode) {
  const tbody = document.getElementById('mode-tbody');
  tbody.innerHTML = '';
  const modes = Object.entries(byMode).sort((a,b) => b[1].count - a[1].count);
  modes.forEach(([mode, v]) => {
    const pct   = Math.max(0, v.avg_reduction_pct);
    const color = pct >= 50 ? '#22c55e' : pct >= 20 ? '#eab308' : '#ef4444';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="mode-pill">${mode}</span></td>
      <td>${v.count}</td>
      <td>${v.avg_before.toLocaleString()}</td>
      <td>${v.avg_after.toLocaleString()}</td>
      <td class="green">${v.avg_saved.toLocaleString()}</td>
      <td style="color:${color}">${v.avg_reduction_pct > 0 ? '-' : '+'}${Math.abs(v.avg_reduction_pct)}%</td>
      <td><div class="bar-wrap"><div class="bar" style="width:${Math.min(100,pct)}%;background:${color}"></div></div></td>
    `;
    tbody.appendChild(tr);
  });
  if (!modes.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="color:var(--muted);padding:16px 10px">No requests yet</td></tr>';
  }
}

let _ctxData = {raw: '', transformed: ''};
let _ctxView  = 'raw';

function switchView(which) {
  _ctxView = which;
  document.getElementById('modal-ctx').textContent = _ctxData[which] || '(empty)';
  document.getElementById('btn-raw').style.opacity         = which === 'raw'         ? '1' : '0.45';
  document.getElementById('btn-transformed').style.opacity = which === 'transformed' ? '1' : '0.45';
  document.getElementById('modal-title').textContent =
    which === 'raw' ? 'Original Context' : 'Modified Context (wrapper output)';
}

async function openCtx(reqId, mode, model) {
  const overlay = document.getElementById('modal-overlay');
  document.getElementById('modal-meta').textContent = `${reqId}  \\u00b7  mode: ${mode}  \\u00b7  model: ${model}`;
  document.getElementById('modal-ctx').textContent  = 'loading...';
  _ctxData = {raw: '', transformed: ''};
  overlay.classList.add('open');
  switchView('raw');
  try {
    const r = await fetch(`/v1/requests/${reqId}/context`);
    const d = await r.json();
    _ctxData.raw         = d.raw         || '(empty)';
    _ctxData.transformed = d.transformed || '(empty)';
    document.getElementById('modal-ctx').textContent = _ctxData[_ctxView];
  } catch(e) {
    document.getElementById('modal-ctx').textContent = 'Error loading context.';
  }
}

function closeModal(e) {
  if (e && e.target !== document.getElementById('modal-overlay')) return;
  document.getElementById('modal-overlay').classList.remove('open');
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') document.getElementById('modal-overlay').classList.remove('open');
});

function addFeedRow(e) {
  const container = document.getElementById('feed-rows');
  const pctColor  = e.reduction_pct >= 50 ? '#22c55e' : e.reduction_pct >= 20 ? '#eab308' : '#ef4444';
  const modelShort = (e.model || '').split('/').pop() || e.model || '—';
  const reqId = e.request_id || '';
  const mode  = e.mode || '—';
  const model = e.model || '—';
  const row = document.createElement('div');
  row.className = 'feed-row new';
  row.innerHTML = `
    <span style="color:var(--muted)">${e.ts}</span>
    <span><span class="mode-pill">${mode}</span></span>
    <span style="color:var(--muted);font-size:10px" title="${model}">${modelShort}</span>
    <span>${e.tok_before.toLocaleString()}</span>
    <span>${e.tok_after.toLocaleString()}</span>
    <span class="green">${e.tok_saved.toLocaleString()}</span>
    <span style="color:${pctColor}">${e.reduction_pct > 0 ? '-' : '+'}${Math.abs(e.reduction_pct)}%</span>
    <span>${reqId ? `<button class="btn sm" onclick="openCtx('${reqId}','${mode}','${model.replace(/'/g,"\\\\'")}')">view</button>` : '—'}</span>
  `;
  container.insertBefore(row, container.firstChild);
  while (container.children.length > 50) container.removeChild(container.lastChild);
  document.getElementById('sb-last').textContent =
    `last: ${mode} (${modelShort}) — saved ${e.tok_saved} tokens (${e.reduction_pct > 0 ? '-' : '+'}${Math.abs(e.reduction_pct)}%)`;
}

// SSE
function connect() {
  const es = new EventSource('/v1/stats/stream');
  es.onopen  = () => setConn(true);
  es.onerror = () => { setConn(false); es.close(); setTimeout(connect, 3000); };
  es.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      updateCards(msg.stats);
      updateModeTable(msg.stats.by_mode);
      if (msg.latest) addFeedRow(msg.latest);
    } catch(e) {}
  };
}

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
    print(f"  Mode: {CONTEXT_MODE} | Auto: {AUTO_MODE} | Upstream: {UPSTREAM_URL or '(not set)'}\n")
    uvicorn.run("wrapper_server:app", host=HOST, port=PORT, reload=False)
