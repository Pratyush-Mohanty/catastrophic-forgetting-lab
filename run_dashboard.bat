@echo off
REM Launch the Catastrophic Forgetting Lab dashboard with the correct (venv) interpreter.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Create it first:
    echo     py -3.10 -m venv .venv
    echo     .venv\Scripts\Activate.ps1
    echo     pip install -r requirements.txt
    pause
    exit /b 1
)
echo Starting dashboard at http://localhost:8502
".venv\Scripts\python.exe" -m streamlit run app/dashboard.py --server.headless=true --server.port=8502
