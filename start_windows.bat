@echo off
cd /d "%~dp0"
echo Installing packages (first time only takes a minute)...
python -m pip install -r requirements.txt
echo.
echo VibeGuard is starting. Open http://localhost:8000 in your browser.
echo Press Ctrl+C in this window to stop it.
python -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
