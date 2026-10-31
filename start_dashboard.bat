@echo off
setlocal
cd /d "%~dp0"

start "Vibration ML Python Backend" cmd /k python backend\app.py
timeout /t 2 /nobreak >nul
start "Vibration ML React Website" cmd /k "cd /d frontend && npm run dev"

echo Started the Python API and React website in separate windows.
echo In the website, click Activate backend to start the sensor logger.
echo Use the N menu and Stop all servers to close the services later.
endlocal
