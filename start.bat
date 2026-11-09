@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   Auto Crop for Scans - local launcher
echo ============================================

python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo Failed to install dependencies. Make sure Python is on PATH.
    pause
    exit /b 1
)

start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8000/"

echo Starting server; the browser will open http://127.0.0.1:8000/ automatically.
echo Close this window to stop the server.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

pause
