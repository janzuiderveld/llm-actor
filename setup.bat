@echo off

python -m venv venv
call venv\Scripts\activate
pip install -e .

REM Copy .env only if it does not exist
if not exist ".env" (
    copy ".env.example" ".env"
    notepad ".env"
)

python -m sounddevice

pause