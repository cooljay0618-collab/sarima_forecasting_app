@echo off
   chcp 65001 > nul
   cd /d "%~dp0"
   python -m streamlit run sarima_forecasting_app.py
   pause