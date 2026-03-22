@echo off
REM ============================================================
REM  setup.bat — Windows one-click project setup
REM  Run this ONCE from your project folder:
REM    cd C:\Users\<you>\Desktop\air_quality_project
REM    setup.bat
REM ============================================================

echo.
echo ======================================================
echo  Nairobi Air Quality Project — Windows Setup
echo ======================================================
echo.

REM Check Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [4/5] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed. Check your internet connection.
    pause
    exit /b 1
)

echo [5/5] Checking .env file...
if not exist .env (
    copy .env.example .env
    echo [WARNING] .env created from template. 
    echo           DATA_URL is already set — edit .env if you need to change it.
) else (
    echo [OK] .env already exists.
)

echo.
echo ======================================================
echo  Setup complete!
echo.
echo  To start working:
echo    1. Open this folder in VS Code:
echo         code .
echo    2. Select the venv Python interpreter:
echo         Ctrl+Shift+P -> "Python: Select Interpreter"
echo         Choose: .\venv\Scripts\python.exe
echo    3. Run the notebook first:
echo         Open data_cleaning_eda.ipynb in VS Code
echo    4. Launch the dashboard:
echo         venv\Scripts\activate
echo         streamlit run dashboard.py
echo ======================================================
echo.
pause
