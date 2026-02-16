"""FastAPI dependency injection for session management"""

from fastapi import Request, HTTPException

from backend.sessions import get_session, create_session


def get_current_session(request: Request) -> dict:
    """Get the current session from cookie or auto-created session"""
    from backend.config import SESSION_COOKIE_NAME

    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id:
        session = get_session(session_id)
        if session:
            return session

    # Check if middleware attached a new session
    session_id = getattr(request.state, "new_session_id", None)
    if session_id:
        session = get_session(session_id)
        if session:
            return session

    raise HTTPException(status_code=401, detail="No active session")
