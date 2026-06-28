#!/usr/bin/env bash
# Starts ContextZip and creates the local .venv on first run if needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

cd "$ROOT"

# Find an available Python executable
SYS_PY=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        SYS_PY="$cmd"
        break
    fi
done
if [ -z "$SYS_PY" ]; then
    echo "[ContextZip] ERROR: Python 3.10+ not found. Install Python and try again."
    exit 1
fi

# Create .venv if missing
if [ ! -f "$PY" ]; then
    echo "[ContextZip] Creating local .venv with $SYS_PY ..."
    "$SYS_PY" -m venv .venv
    echo "[ContextZip] Installing dependencies..."
    "$PY" -m pip install --upgrade pip --quiet
    "$PY" -m pip install -r requirements.txt --quiet
    echo "[ContextZip] Dependencies installed."
fi

# Reinstall dependencies if requirements.txt changed
STAMP="$VENV/.installed_stamp"
if [ requirements.txt -nt "$STAMP" ] 2>/dev/null || [ ! -f "$STAMP" ]; then
    echo "[ContextZip] requirements.txt changed - reinstalling dependencies..."
    "$PY" -m pip install -r requirements.txt --quiet
    touch "$STAMP"
fi

# Create .env if missing
if [ ! -f "$ROOT/.env" ]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    echo "[ContextZip] Created .env from .env.example - configure UPSTREAM_API_KEY."
fi

# Start server
echo ""
echo "[ContextZip] Starting server..."
exec "$PY" wrapper_server.py
