# Copyright (c) 2026 Gaetano Marcello Incarbone. MIT License — see LICENSE file.
"""Alfa1 — FastAPI routes and in-memory session state."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import alfa1_tools
from alfa1_agent import run_agent_turn
from alfa1_tools import Alfa1Error
from alfa1_ui import ALFA1_HTML

router = APIRouter()

# In-memory state — reset on process restart, same convention as
# wrapper_server._stats / _recent / _sse_queues.
_conversation: list[dict] = []
_turn_status: str = "idle"   # "idle" | "working" | "ok" | "error"
_sse_queues: list[asyncio.Queue] = []
_current_task: asyncio.Task | None = None   # the in-flight run_agent_turn task, if any


async def _broadcast(evt: dict) -> None:
    payload = json.dumps(evt)
    for q in list(_sse_queues):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


async def _read_json(request: Request) -> tuple[dict | None, JSONResponse | None]:
    """Parse the request body as JSON, returning a 400 JSONResponse instead
    of letting a malformed/truncated body raise an unhandled 500 — mirrors
    wrapper_server.chat_completions' own try/except around request.json()."""
    try:
        return await request.json(), None
    except Exception as exc:
        return None, JSONResponse({"error": f"Invalid JSON body: {exc}"}, status_code=400)


async def _activate_workspace(path: str) -> dict:
    """Set the workspace and load that folder's own persisted conversation
    (<root>/.alfa1/history.json), if any — used both when the user picks a
    folder and when auto-restoring the last-used one on startup, so a server
    restart (or the tab just being closed and reopened) doesn't silently
    lose in-progress work the way plain in-memory-only state would."""
    global _turn_status
    await _cancel_current_task()
    result = alfa1_tools.set_workspace(path)
    history = alfa1_tools.load_history()
    _conversation.clear()
    if history:
        _conversation.extend(history)
    _turn_status = "idle"
    return result


@router.get("", response_class=HTMLResponse)
async def alfa1_page():
    return HTMLResponse(ALFA1_HTML)


@router.get("/workspace")
async def get_workspace_route():
    root = alfa1_tools.get_workspace()
    if root is None:
        last = alfa1_tools.get_last_workspace()
        if last is not None:
            await _activate_workspace(str(last))
            root = alfa1_tools.get_workspace()
    return {"root": str(root) if root else None}


@router.post("/workspace")
async def set_workspace_route(request: Request):
    body, err = await _read_json(request)
    if err:
        return err
    try:
        result = await _activate_workspace(body.get("path", ""))
    except Alfa1Error as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return result


@router.post("/workspace/pick")
async def pick_workspace_route():
    try:
        path = await asyncio.to_thread(alfa1_tools.pick_workspace_dialog)
    except Exception as exc:  # tkinter/display errors — surface, don't crash the server
        return JSONResponse({"error": str(exc)}, status_code=500)
    if not path:
        return {"root": None, "cancelled": True}
    result = await _activate_workspace(path)
    return result


@router.get("/files/tree")
async def files_tree(path: str = "."):
    try:
        entries = alfa1_tools.list_tree(path)
    except Alfa1Error as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"entries": entries}


@router.get("/files/content")
async def files_read(path: str):
    try:
        return alfa1_tools.read_file(path)
    except Alfa1Error as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@router.put("/files/content")
async def files_write(request: Request):
    body, err = await _read_json(request)
    if err:
        return err
    try:
        result = alfa1_tools.write_file(body["path"], body.get("content", ""))
    except (Alfa1Error, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    await _broadcast({"type": "file_changed", "path": result["path"]})
    return result


@router.delete("/files/content")
async def files_delete(path: str):
    try:
        result = alfa1_tools.delete_file(path)
    except Alfa1Error as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    await _broadcast({"type": "file_changed", "path": path})
    return result


@router.post("/reset")
async def reset_conversation():
    """"New task" reset: clears the current conversation and starts fresh,
    but leaves an empty history.json in place (via save_history below) —
    the next turn resumes normal persistence immediately."""
    global _turn_status
    await _cancel_current_task()
    _conversation.clear()
    _turn_status = "idle"
    alfa1_tools.save_history(_conversation)
    await _broadcast({"type": "snapshot", "conversation": _conversation, "status": _turn_status})
    return {"ok": True}


@router.delete("/history")
async def clear_all_sessions():
    """"Clear all stored sessions": a harder wipe than /reset — removes the
    persisted history.json file for the current workspace entirely, not
    just the in-memory conversation."""
    global _turn_status
    await _cancel_current_task()
    _conversation.clear()
    _turn_status = "idle"
    deleted = alfa1_tools.delete_history()
    await _broadcast({"type": "snapshot", "conversation": _conversation, "status": _turn_status})
    return {"ok": True, "deleted": deleted}


@router.post("/cancel")
async def cancel_turn():
    """Force-stop a stuck turn (e.g. the agent ran a command that blocks
    waiting on stdin, or an upstream call is hanging) without discarding the
    conversation history the way /reset does."""
    global _turn_status
    cancelled = await _cancel_current_task()
    _turn_status = "idle"
    await _broadcast({"type": "status", "status": "idle"})
    return {"cancelled": cancelled}


async def _cancel_current_task() -> bool:
    global _current_task
    if _current_task is not None and not _current_task.done():
        _current_task.cancel()
        try:
            await _current_task
        except (asyncio.CancelledError, Exception):
            pass
        _current_task = None
        return True
    _current_task = None
    return False


@router.post("/chat")
async def post_chat(request: Request):
    global _turn_status
    if _turn_status == "working":
        return JSONResponse({"error": "A turn is already in progress"}, status_code=400)
    if alfa1_tools.get_workspace() is None:
        return JSONResponse({"error": "No workspace selected"}, status_code=400)

    body, err = await _read_json(request)
    if err:
        return err
    message = body.get("message", "")
    model = body.get("model")

    _conversation.append({"role": "user", "content": message})
    _turn_status = "working"
    await _broadcast({"type": "status", "status": "working"})

    async def on_event(evt: dict) -> None:
        await _broadcast(evt)

    async def _run() -> None:
        global _turn_status
        try:
            await run_agent_turn(_conversation, model=model, on_event=on_event)
            _turn_status = "ok"
            await _broadcast({"type": "status", "status": "ok"})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface any unexpected failure to the UI
            _turn_status = "error"
            await _broadcast({"type": "status", "status": "error"})
            await _broadcast({"type": "error", "message": str(exc)})
        finally:
            alfa1_tools.save_history(_conversation)

    global _current_task
    _current_task = asyncio.create_task(_run())
    return {"accepted": True}


@router.get("/stream")
async def alfa1_stream():
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _sse_queues.append(q)

    async def event_gen() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'snapshot', 'conversation': _conversation, 'status': _turn_status})}\n\n"
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
