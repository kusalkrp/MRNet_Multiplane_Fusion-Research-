@echo off
echo ============================================================
echo  Knee MRI Research Dashboard - PyTorch Inference Engine
echo  K.R. Punchihewa - Coventry University
echo ============================================================
echo.
echo Activating GPU Environment...
echo Launching Streamlit App...
echo.

cd /d "%~dp0"

IF EXIST "C:\Users\ganee\.conda\envs\gpu\python.exe" (
    "C:\Users\ganee\.conda\envs\gpu\python.exe" -m streamlit run app.py
) ELSE (
    call conda activate gpu
    streamlit run app.py
)

pause
