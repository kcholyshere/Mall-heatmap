#!/usr/bin/env bash
# Double-click launcher (macOS): sets up a local environment and starts the app.
# Requires Python 3.11+ installed. First run installs dependencies (slow); later runs are quick.
cd "$(dirname "$0")" || exit 1

if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "Installing dependencies (first run only - please wait)..."
pip install --quiet --requirement requirements.txt

echo "Starting Mall Heatmap - your browser will open at http://localhost:8501"
streamlit run app.py
