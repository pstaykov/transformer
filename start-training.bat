@echo off
REM Double-click to resume training with the exact same settings as before
REM (run3, step ~313k) - all parameters already live in docker-compose.yml,
REM nothing to type here.
cd /d "%~dp0"
docker compose up train
pause
