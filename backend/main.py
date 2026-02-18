"""
SCOUT Web Backend
FastAPI application serving the DoE Suite API
"""

import asyncio
import logging
import sys
import os
from contextlib import asynccontextmanager

# Configure logging to show debug info from backend modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure project root is in path so core/ imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import CORS_ORIGINS, SESSION_COOKIE_NAME, SESSION_MAX_AGE
from backend.sessions import get_session, create_session, cleanup_expired_sessions

SESSION_HEADER = "X-Session-ID"

from backend.routers import config_routes, project, design, analysis


async def _session_cleanup_loop():
    """Periodically clean expired sessions"""
    logger = logging.getLogger("backend.session")
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        removed = cleanup_expired_sessions(SESSION_MAX_AGE)
        if removed:
            logger.info(f"[SESSION] Cleaned {removed} expired session(s)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown tasks"""
    task = asyncio.create_task(_session_cleanup_loop())
    yield
    task.cancel()
    from backend.sessions import _sessions
    _sessions.clear()


app = FastAPI(
    title="SCOUT API",
    description="Screening & Condition Optimization Utility Tool - Web API",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS middleware for SvelteKit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[SESSION_HEADER],
)

@app.middleware("http")
async def auto_session_middleware(request: Request, call_next):
    """Auto-create a session if none exists (header or cookie)"""
    # Check header first, then cookie
    session_id = request.headers.get(SESSION_HEADER)
    if session_id and get_session(session_id):
        return await call_next(request)

    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id and get_session(session_id):
        return await call_next(request)

    # Auto-create for API requests (skip read-only config/health endpoints)
    if request.url.path.startswith("/api/") and not request.url.path.startswith(("/api/config", "/api/health")):
        session_id = create_session("SCOUT Project")
        request.state.new_session_id = session_id
        logging.getLogger("backend.session").info(
            f"[SESSION] New session {session_id[:8]} for {request.url.path}"
        )

    response: Response = await call_next(request)

    # Return session ID in header so frontend can store it
    if hasattr(request.state, "new_session_id"):
        response.headers[SESSION_HEADER] = request.state.new_session_id

    return response


# Include routers
app.include_router(config_routes.router, prefix="/api/config", tags=["Config"])
app.include_router(project.router, prefix="/api/project", tags=["Project"])
app.include_router(design.router, prefix="/api/design", tags=["Design"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "version": "0.2.0"}


# Serve static frontend if build directory exists (Docker/production)
STATIC_DIR = os.path.join(PROJECT_ROOT, "frontend", "build")
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
