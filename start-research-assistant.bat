@echo off
REM One-click start (PRODUCTION mode) — opens the browser research UI
REM served by the local FastAPI server (no Vite, no Node needed at runtime).
REM Safe to run twice: a second run detects the running app and just
REM opens the browser. See docs\OPERATIONS.md for troubleshooting.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launcher.ps1" -Action start
