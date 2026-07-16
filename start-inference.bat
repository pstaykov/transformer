@echo off
REM Double-click to start the showcase site + chat server at http://localhost:8000
cd /d "%~dp0"
docker compose up inference
pause
