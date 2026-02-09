@echo off
REM ============================================
REM Protein Stability DoE Suite Launcher
REM Double-click this file to run the application
REM ============================================

echo ==========================================
echo   Protein Stability DoE Suite
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found: %PYTHON_VERSION%

REM Virtual environment directory
set VENV_DIR=.venv

REM Check if virtual environment exists
if not exist "%VENV_DIR%\" (
    echo.
    echo First run detected. Setting up environment...
    echo Creating virtual environment...
    python -m venv %VENV_DIR%

    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
)

REM Activate virtual environment
call %VENV_DIR%\Scripts\activate.bat

REM Check if dependencies are installed
set DEPS_MARKER=%VENV_DIR%\.deps_installed

if not exist "%DEPS_MARKER%" (
    echo.
    echo Installing dependencies (this may take a few minutes^)...
    pip install --upgrade pip -q
    pip install -r requirements.txt

    if %errorlevel% neq 0 (
        echo Error: Failed to install dependencies.
        pause
        exit /b 1
    )

    REM Create marker file
    echo. > %DEPS_MARKER%
    echo Dependencies installed successfully.
)

REM Run the application
echo.
echo Launching application...
echo.
python main.py

REM Deactivate virtual environment
deactivate
