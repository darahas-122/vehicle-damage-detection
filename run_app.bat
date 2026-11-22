@echo off
title VehDetect AI - Vehicle Damage Detection
echo ============================================================
echo   Starting VehDetect AI Server & Live Public Link
echo ============================================================
echo.
cd /d "%~dp0"
.\.venv\python.exe run_with_tunnel.py
pause
