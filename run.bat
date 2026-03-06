@echo off
echo Starting CoffeeAI System...
if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe main.py --serve
) else (
    echo [ERROR] Virtual environment not found! 
    echo Please make sure you are in the project root and .venv folder exists.
    pause
)
