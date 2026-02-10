@echo off
REM ============================================
REM Protein Stability DoE Suite Launcher
REM Double-click this file to run the application
REM ============================================

echo ==========================================
echo   Protein Stability DoE Suite
echo ==========================================
echo.

REM Check if Python is installed (try python first, then python3)
set PYTHON_CMD=python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    set PYTHON_CMD=python3
    python3 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: Python is not installed or not in PATH.
        echo.
        echo Please install Python 3.10 or newer from:
        echo   https://www.python.org/downloads/
        echo.
        echo IMPORTANT: During installation, check the box that says
        echo   "Add Python to PATH"
        echo.
        pause
        exit /b 1
    )
)

for /f "tokens=*" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found: %PYTHON_VERSION%

REM Virtual environment directory
set VENV_DIR=.venv

REM Check if virtual environment exists
if not exist "%VENV_DIR%\" (
    echo.
    echo First run detected. Setting up environment...
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv %VENV_DIR%

    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment.
        echo Make sure the 'venv' module is available for your Python installation.
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
    echo Installing dependencies...
    echo This includes PyTorch and other large packages.
    echo Please be patient, this may take 5-15 minutes on first run.
    echo The install may appear to pause - this is normal.
    echo.
    python -m pip install --upgrade pip -q 2>nul
    python -m pip install -r requirements.txt

    if %errorlevel% neq 0 (
        echo.
        echo Error: Failed to install dependencies.
        echo Check your internet connection and try again.
        echo If the problem persists, try deleting the .venv folder and re-running.
        pause
        exit /b 1
    )

    REM Create marker file
    echo. > %DEPS_MARKER%
    echo.
    echo Dependencies installed successfully.
)

REM Run the application
echo.
echo Launching application...
echo.
python main.py

REM Keep window open if app exits with error
if %errorlevel% neq 0 (
    echo.
    echo Application exited with an error.
    pause
)

REM Deactivate virtual environment
deactivate
