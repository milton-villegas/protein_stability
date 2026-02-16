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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in path so core/ imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import CORS_ORIGINS
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
    version="2.0.0",
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

# Include routers
app.include_router(config_routes.router, prefix="/api/config", tags=["Config"])
app.include_router(project.router, prefix="/api/project", tags=["Project"])
app.include_router(design.router, prefix="/api/design", tags=["Design"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "version": "2.0.0"}
