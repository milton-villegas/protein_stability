"""
SCOUT Web Backend
FastAPI application serving the DoE Suite API
"""

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

from backend.config import CORS_ORIGINS, SESSION_COOKIE_NAME
from backend.sessions import get_session, create_session
from backend.routers import config_routes, project, design, analysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown tasks"""
    yield
    # Cleanup sessions on shutdown
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
)

@app.middleware("http")
async def auto_session_middleware(request: Request, call_next):
    """Auto-create a session if no valid cookie exists"""
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    has_valid_session = session_id and get_session(session_id) is not None

    if not has_valid_session and request.url.path.startswith("/api/"):
        session_id = create_session("SCOUT Project")
        request.state.new_session_id = session_id

    response: Response = await call_next(request)

    # Set cookie on new sessions
    if hasattr(request.state, "new_session_id"):
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=request.state.new_session_id,
            httponly=True,
            samesite="none",
            secure=True,
            max_age=3600,
        )

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
