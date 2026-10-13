@echo off
REM Stops ONLY the processes this launcher started (verified by PID file
REM + command-line marker). Never touches unrelated programs, never
REM touches your saved research in data\.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launcher.ps1" -Action stop
