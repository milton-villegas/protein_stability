"""In-memory session store for managing per-user project state"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from core.project import DoEProject
from core.doe_designer import DoEDesigner
from core.doe_analyzer import DoEAnalyzer
from core.design_factory import DesignFactory
from core.plotter import DoEPlotter
from core.exporter import ResultsExporter

# Global session store: session_id -> session_data
_sessions: Dict[str, Dict[str, Any]] = {}


def create_session(project_name: str = "Untitled Project") -> str:
    """Create a new session with a fresh DoEProject and return session_id"""
    session_id = str(uuid.uuid4())

    # Check for optional libraries
    has_pydoe3 = False
    has_smt = False
    try:
        import pyDOE3
        has_pydoe3 = True
    except ImportError:
        pass
    try:
        import smt
        has_smt = True
    except ImportError:
        pass

    project = DoEProject()
    project.name = project_name

    _sessions[session_id] = {
        "project": project,
        "designer": DoEDesigner(),
        "analyzer": DoEAnalyzer(),
        "design_factory": DesignFactory(has_pydoe3=has_pydoe3, has_smt=has_smt),
        "plotter": DoEPlotter(),
        "exporter": ResultsExporter(),
        "optimizer": None,
        "data_handler": None,
        "has_pydoe3": has_pydoe3,
        "has_smt": has_smt,
        "last_accessed": datetime.now(),
    }

    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session data by ID, updating last_accessed"""
    session = _sessions.get(session_id)
    if session:
        session["last_accessed"] = datetime.now()
    return session


def delete_session(session_id: str) -> None:
    """Remove a session"""
    _sessions.pop(session_id, None)


def cleanup_expired_sessions(max_age_seconds: int = 3600) -> int:
    """Remove sessions older than max_age_seconds. Returns count removed."""
    now = datetime.now()
    expired = [
        sid for sid, data in _sessions.items()
        if (now - data["last_accessed"]).total_seconds() > max_age_seconds
    ]
    for sid in expired:
        del _sessions[sid]
    return len(expired)
