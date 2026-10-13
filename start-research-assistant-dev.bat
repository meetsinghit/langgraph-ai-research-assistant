@echo off
REM Development start — backend plus the Vite dev server (hot reload).
REM Use start-research-assistant.bat instead for normal daily research.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launcher.ps1" -Action dev
