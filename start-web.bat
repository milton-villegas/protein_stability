@echo off
echo ======================================
echo   SCOUT Web Application
echo ======================================
echo.

echo Starting backend on http://localhost:8000...
set PYTHONPATH=%~dp0
start /B cmd /c "cd %~dp0 && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 2 /nobreak > nul

echo Starting frontend on http://localhost:5173...
start /B cmd /c "cd %~dp0\frontend && npm run dev -- --host 0.0.0.0 --port 5173"

echo.
echo ======================================
echo   SCOUT is running!
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo ======================================
echo.
echo Press any key to stop...
pause > nul
taskkill /f /im uvicorn.exe 2>nul
taskkill /f /im node.exe 2>nul
