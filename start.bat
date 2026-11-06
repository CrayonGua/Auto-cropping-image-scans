@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   扫描件自动裁剪 - 本地启动
echo ============================================

python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo 依赖安装失败，请检查 Python 是否已加入 PATH。
    pause
    exit /b 1
)

start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8000/"

echo 服务启动中，浏览器将自动打开 http://127.0.0.1:8000/
echo 关闭本窗口即可停止服务。
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

pause
