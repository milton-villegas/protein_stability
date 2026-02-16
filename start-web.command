#!/bin/bash
# Start SCOUT Web Application (Backend + Frontend)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "======================================"
echo "  SCOUT Web Application"
echo "======================================"
echo ""

# Start FastAPI backend
echo "Starting backend on http://localhost:8000..."
cd "$SCRIPT_DIR"
PYTHONPATH="$SCRIPT_DIR" python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend
sleep 2

# Start SvelteKit frontend
echo "Starting frontend on http://localhost:5173..."
cd "$SCRIPT_DIR/frontend"
VITE_API_URL="http://localhost:8000" npm run dev -- --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!

echo ""
echo "======================================"
echo "  SCOUT is running!"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "======================================"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
