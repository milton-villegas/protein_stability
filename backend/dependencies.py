"""FastAPI dependency injection for session management"""

from fastapi import Request, HTTPException

from backend.sessions import get_session, create_session

# Single shared session for simple deployments (HF Spaces, single-user)
_default_session_id = None


def get_current_session(request: Request) -> dict:
    """Get the current session, auto-creating one if needed"""
    global _default_session_id
    from backend.config import SESSION_COOKIE_NAME

    # Try cookie-based session first
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id:
        session = get_session(session_id)
        if session:
            return session

    # Fall back to default shared session
    if _default_session_id:
        session = get_session(_default_session_id)
        if session:
            return session

    # Auto-create a default session
    _default_session_id = create_session("SCOUT Project")
    return get_session(_default_session_id)
