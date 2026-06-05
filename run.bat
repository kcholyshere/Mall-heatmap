@echo off
REM Double-click launcher (Windows): sets up a local environment and starts the app.
REM Requires Python 3.11+ installed. First run installs dependencies (slow); later runs are quick.
cd /d "%~dp0"

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
echo Installing dependencies (first run only - please wait)...
pip install --quiet --requirement requirements.txt

echo Starting Mall Heatmap - your browser will open at http://localhost:8501
streamlit run app.py
