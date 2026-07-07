# Copyright (c) 2026 Gaetano Marcello Incarbone. MIT License — see LICENSE file.
"""Alfa1 — workspace-scoped file I/O and shell command execution tools.

No allowlist/blacklist on shell commands: scoping the working directory to a
chosen workspace root, plus a timeout and output-size cap, is the only safety
boundary. That is intentional for this local, single-user dev tool — the
user has full permissions on the folder they picked, by design. Only the
`cwd` a command runs in is containment-checked; the command text itself is
free-form shell syntax.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from pathlib import Path, PurePosixPath

_SESSION_DIR_NAME = ".alfa1"
_SKIP_DIR_NAMES = {".git", "node_modules", "__pycache__", ".venv", "venv",
                   ".mypy_cache", ".pytest_cache", _SESSION_DIR_NAME}

# Where the last-used workspace path is remembered — lives next to the
# shapeshifter installation (same convention as wrapper_server's
# .shapeshifter_keys.json), NOT inside any workspace, since we need it
# before a workspace is even chosen. Module-level so tests can monkeypatch
# it to a tmp_path and never touch the real user's pointer file.
#
# Keyed by WRAPPER_PORT: running a second instance on a different port (e.g.
# ad hoc manual testing alongside a real running session) must never clobber
# the real session's remembered workspace — observed in practice when a
# throwaway instance on another port overwrote the pointer a live session on
# 8787 auto-restored on its next restart, silently swapping in the wrong
# project folder.
_LAST_WORKSPACE_PATH = Path(__file__).parent / f".alfa1_last_workspace_{os.getenv('WRAPPER_PORT', '8787')}.json"


class Alfa1Error(Exception):
    """Raised for any invalid workspace/path/file operation in Alfa1."""


_workspace_root: Path | None = None


def set_workspace(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise Alfa1Error(f"Not a directory: {path!r}")
    global _workspace_root
    _workspace_root = p
    (p / _SESSION_DIR_NAME).mkdir(exist_ok=True)
    try:
        _LAST_WORKSPACE_PATH.write_text(json.dumps({"root": str(p)}), encoding="utf-8")
    except OSError:
        pass  # remembering the path is a convenience, not a correctness requirement
    return {"root": str(p)}


def get_workspace() -> Path | None:
    return _workspace_root


def get_last_workspace() -> Path | None:
    """The most recently used workspace, if its pointer file exists and the
    directory it names still exists — used to restore a session across a
    server restart without asking the user to re-pick the folder."""
    try:
        data = json.loads(_LAST_WORKSPACE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    root = data.get("root")
    if not root:
        return None
    p = Path(root)
    return p if p.is_dir() else None


def _session_file() -> Path:
    return _workspace_root / _SESSION_DIR_NAME / "history.json"


def save_history(conversation: list[dict]) -> None:
    """Persist the chat conversation to <workspace>/.alfa1/history.json so
    it survives a server restart. Best-effort: a write failure here should
    never break the agent turn that triggered it."""
    if _workspace_root is None:
        return
    try:
        _session_file().write_text(json.dumps(conversation, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def delete_history() -> bool:
    """Remove the persisted history file for the current workspace entirely
    (as opposed to save_history([]), which leaves an empty-array file in
    place) — used by the "clear all stored sessions" UI action, a harder
    wipe than the "new task" soft reset. Returns whether a file actually
    existed to delete."""
    if _workspace_root is None:
        return False
    f = _session_file()
    if not f.exists():
        return False
    try:
        f.unlink()
        return True
    except OSError:
        return False


def load_history() -> list[dict] | None:
    """Previously saved conversation for the current workspace, or None if
    there isn't one (a brand-new workspace, or one never used with Alfa1)."""
    if _workspace_root is None:
        return None
    f = _session_file()
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def pick_workspace_dialog() -> str | None:
    """Blocking native folder picker (tkinter). Must be called via
    asyncio.to_thread — never directly inside an async route handler, or it
    will freeze the whole event loop until the dialog is dismissed."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askdirectory(title="Select Alfa1 workspace folder")
    finally:
        root.destroy()
    return path or None


def _safe_join(rel_path: str | None) -> Path:
    """Resolve rel_path against the workspace root; raise Alfa1Error if the
    result would escape the root (absolute paths, '..' traversal, drive
    letters, or symlinks pointing outside)."""
    if _workspace_root is None:
        raise Alfa1Error("No workspace set")
    if not rel_path:
        rel_path = "."
    normalized = rel_path.replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        raise Alfa1Error(f"Absolute paths not allowed: {rel_path!r}")
    if PurePosixPath(normalized).is_absolute():
        raise Alfa1Error(f"Absolute paths not allowed: {rel_path!r}")
    root = _workspace_root.resolve()
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise Alfa1Error(f"Path escapes workspace: {rel_path!r}")
    return candidate


def list_tree(rel_path: str = ".", max_entries: int = 2000) -> list[dict]:
    base = _safe_join(rel_path)
    if not base.is_dir():
        raise Alfa1Error(f"Not a directory: {rel_path!r}")
    root = _workspace_root.resolve()
    entries: list[dict] = []

    def _walk(d: Path) -> None:
        try:
            children = sorted(d.iterdir(), key=lambda c: (c.is_file(), c.name.lower()))
        except OSError:
            return
        for c in children:
            if len(entries) >= max_entries:
                return
            if c.name in _SKIP_DIR_NAMES:
                continue
            rel = c.resolve().relative_to(root).as_posix()
            if c.is_dir():
                entries.append({"path": rel, "type": "dir", "size": None})
                _walk(c)
            else:
                try:
                    size = c.stat().st_size
                except OSError:
                    size = None
                entries.append({"path": rel, "type": "file", "size": size})

    _walk(base)
    return entries


def search_files(query: str, rel_path: str = ".", max_results: int = 200,
                  case_sensitive: bool = False) -> list[dict]:
    """Text search across files under rel_path. `query` is tried as a regex
    first and falls back to a literal substring search if it isn't valid
    regex syntax — lets a plain word search work without the caller needing
    to know regex escaping rules, while still allowing regex when wanted."""
    base = _safe_join(rel_path)
    if not base.is_dir():
        raise Alfa1Error(f"Not a directory: {rel_path!r}")
    root = _workspace_root.resolve()
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(query, flags)
    except re.error:
        pattern = re.compile(re.escape(query), flags)

    results: list[dict] = []

    def _walk(d: Path) -> None:
        try:
            children = sorted(d.iterdir(), key=lambda c: (c.is_file(), c.name.lower()))
        except OSError:
            return
        for c in children:
            if len(results) >= max_results:
                return
            if c.name in _SKIP_DIR_NAMES:
                continue
            if c.is_dir():
                _walk(c)
                continue
            try:
                raw = c.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:8192]:
                continue  # skip binary files
            text = raw.decode("utf-8", errors="replace")
            rel = c.resolve().relative_to(root).as_posix()
            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    results.append({"path": rel, "line": line_no, "text": line.strip()[:300]})
                    if len(results) >= max_results:
                        return

    _walk(base)
    return results


def read_file(rel_path: str, max_bytes: int = 2_000_000) -> dict:
    target = _safe_join(rel_path)
    if not target.is_file():
        raise Alfa1Error(f"Not a file: {rel_path!r}")
    raw = target.read_bytes()
    if b"\x00" in raw[:8192]:
        return {"path": rel_path, "content": None, "binary": True, "size": len(raw)}
    truncated = len(raw) > max_bytes
    text = raw[:max_bytes].decode("utf-8", errors="replace")
    return {"path": rel_path, "content": text, "truncated": truncated, "size": len(raw), "binary": False}


def write_file(rel_path: str, content: str, create_dirs: bool = True) -> dict:
    target = _safe_join(rel_path)
    if create_dirs:
        target.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    target.write_bytes(data)
    return {"path": rel_path, "bytes_written": len(data)}


def delete_file(rel_path: str) -> dict:
    if rel_path in (".", "", None):
        raise Alfa1Error("Refusing to delete the workspace root")
    target = _safe_join(rel_path)
    if not target.exists():
        raise Alfa1Error(f"Not found: {rel_path!r}")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"path": rel_path, "deleted": True}


async def run_command(command: str, cwd_rel: str = ".", timeout_s: int = 30,
                       max_output_bytes: int = 200_000) -> dict:
    timeout_s = min(timeout_s, 300)
    cwd = _safe_join(cwd_rel)
    if not cwd.is_dir():
        raise Alfa1Error(f"cwd is not a directory: {cwd_rel!r}")

    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_shell(
        command, cwd=str(cwd),
        # Never inherit the server's own stdin: a command that calls input()
        # (e.g. the agent running a script it just wrote) would otherwise
        # block forever waiting on a keystroke nobody can send, tying up the
        # turn until the timeout — and the model tends to retry the same
        # blocking command on failure, compounding it. DEVNULL makes such a
        # read fail fast with EOFError instead.
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    timed_out = False
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        await proc.wait()
        stdout_b, stderr_b = b"", b""
    duration_ms = (time.monotonic() - t0) * 1000

    def _cap(b: bytes) -> tuple[str, bool]:
        truncated = len(b) > max_output_bytes
        return b[:max_output_bytes].decode("utf-8", errors="replace"), truncated

    stdout, out_trunc = _cap(stdout_b)
    stderr, err_trunc = _cap(stderr_b)
    return {
        "exit_code": proc.returncode if not timed_out else None,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": out_trunc or err_trunc,
        "timed_out": timed_out,
        "duration_ms": round(duration_ms, 1),
    }
