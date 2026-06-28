@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV=%SCRIPT_DIR%.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"

cd /d "%SCRIPT_DIR%"

:: Create .venv if missing
if not exist "%VENV%\Scripts\activate.bat" (
    echo [ContextZip] Creating local .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: python not found. Install Python 3.11+ and try again.
        pause
        exit /b 1
    )
    echo [ContextZip] Installing dependencies...
    "%PYTHON%" -m pip install --upgrade pip --quiet
    "%PYTHON%" -m pip install -r requirements.txt --quiet
    echo [ContextZip] Dependencies installed.
)

:: Create .env if missing
if not exist "%SCRIPT_DIR%.env" (
    echo [ContextZip] Copying .env.example to .env ...
    copy "%SCRIPT_DIR%.env.example" "%SCRIPT_DIR%.env" >nul
    echo [ContextZip] Configure UPSTREAM_API_KEY in .env before using the wrapper.
)

:: Start server
echo [ContextZip] Starting server...
echo.
"%PYTHON%" wrapper_server.py
pause
