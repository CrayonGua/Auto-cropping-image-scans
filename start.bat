@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   Auto Crop for Scans - Local Launcher
echo ============================================

if not exist ".venv\Scripts\python.exe" (
    where python >nul 2>&1
    if errorlevel 1 (
        echo Python was not found. Install Python 3.10 or later and add it to PATH.
        pause
        exit /b 1
    )
    echo Creating an isolated Python environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Could not create the virtual environment.
        pause
        exit /b 1
    )
    echo Installing project dependencies...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. Check your internet connection and Python setup.
        pause
        exit /b 1
    )
)

start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8000/"
echo Starting the local service at http://127.0.0.1:8000/
echo Close this window or press Ctrl+C to stop the service.
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

pause
