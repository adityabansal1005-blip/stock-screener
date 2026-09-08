@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run SETUP.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" main.py
pause
