@echo off
rem Start the game: backend + frontend, then open the browser.
rem Double-click this file. Close the window (or press Ctrl+C) to stop.
title WS-FB game
set "ROOT=%~dp0"
set "PY=%ROOT%backend\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Backend venv not found at %PY%
    echo Create it first: python -m venv backend\venv  and  pip install -r backend\requirements.txt
    pause
    exit /b 1
)
"%PY%" -u "%ROOT%run_game.py"
