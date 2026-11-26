@echo off
echo ============================================================
echo   RAG Enterprise Knowledge System - Streamlit Launcher
echo ============================================================
echo.
echo [1/2] Starting FastAPI backend on port 8000...
start "RAG FastAPI Backend" cmd /k ".venv\Scripts\uvicorn api:app --reload --port 8000"

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak > nul

echo [2/2] Starting Streamlit frontend...
echo.
echo  Open in browser: http://localhost:8501
echo.
.venv\Scripts\streamlit run streamlit_app.py --server.port 8501 --server.headless false
