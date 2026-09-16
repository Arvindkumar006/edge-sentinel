@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM Edge Sentinel - Launcher
REM Resolves environment, checks runtime providers, and launches Edge Sentinel
REM without altering configuration or duplicating backend-selection logic.
REM ============================================================================

REM 1. Resolve project directory
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

REM 2. Locate Python executable with priority:
REM    1. <project>\.venv\Scripts\python.exe (if functional)
REM    2. %USERPROFILE%\.venv\Scripts\python.exe (if functional)
REM    3. python.exe from PATH
REM    4. Fallback to any detected .venv
set "PYTHON_CMD="

if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    "%PROJECT_DIR%\.venv\Scripts\python.exe" -c "import yaml" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "PYTHON_CMD=%PROJECT_DIR%\.venv\Scripts\python.exe"
    )
)

if not defined PYTHON_CMD if exist "%USERPROFILE%\.venv\Scripts\python.exe" (
    "%USERPROFILE%\.venv\Scripts\python.exe" -c "import yaml" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "PYTHON_CMD=%USERPROFILE%\.venv\Scripts\python.exe"
    )
)

if not defined PYTHON_CMD (
    where python.exe >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "PYTHON_CMD=python.exe"
    )
)

if not defined PYTHON_CMD (
    if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
        set "PYTHON_CMD=%PROJECT_DIR%\.venv\Scripts\python.exe"
    ) else if exist "%USERPROFILE%\.venv\Scripts\python.exe" (
        set "PYTHON_CMD=%USERPROFILE%\.venv\Scripts\python.exe"
    )
)

if not defined PYTHON_CMD (
    echo [ERROR] Python was not found.
    echo Please ensure Python is installed and available in PATH, or create a virtual environment at .venv.
    exit /b 1
)

REM 3. Lightweight environment check for user feedback
"%PYTHON_CMD%" -c "import sys; import onnxruntime as ort; sys.exit(0 if 'QNNExecutionProvider' in ort.get_available_providers() else 1)" >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo [STATUS] QNN provider available; launching with configured backend.
) else (
    "%PYTHON_CMD%" -c "import onnxruntime" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo [STATUS] QNN provider unavailable; launching with configured backend.
    ) else (
        echo [STATUS] ONNX Runtime not detected; launching with configured backend.
    )
)

REM 4. Launch Edge Sentinel passing through all CLI arguments
"%PYTHON_CMD%" main.py %*
exit /b %ERRORLEVEL%
