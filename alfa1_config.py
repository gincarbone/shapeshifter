# Copyright (c) 2026 Gaetano Marcello Incarbone. MIT License — see LICENSE file.
"""Alfa1 — tiny standalone config helper.

Deliberately does not import wrapper_server (and is not imported by it at
module scope) to avoid a circular import: alfa1_agent needs the proxy's own
host/port to call /v1/chat/completions over HTTP loopback, but wrapper_server
is the module that wires the alfa1 router in.
"""
from __future__ import annotations

import os


def get_self_base_url() -> str:
    """Base URL for this same ShapeShifter process, for loopback HTTP calls."""
    host = os.getenv("WRAPPER_HOST", "127.0.0.1")
    port = int(os.getenv("WRAPPER_PORT", "8787"))
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{port}"
