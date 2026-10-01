@echo off
rem Double-click to start the HomeMonde demand dashboard. Keep this window open while you use it; close it to stop.
cd /d "%~dp0"
echo Starting dashboard at http://localhost:8501 ...
start "" http://localhost:8501
python -m streamlit run app.py
pause
