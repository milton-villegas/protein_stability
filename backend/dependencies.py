"""FastAPI dependency injection for session management"""

from fastapi import Request, HTTPException

from backend.sessions import get_session


def get_current_session(request: Request) -> dict:
    """Get the current session from the request cookie"""
    from backend.config import SESSION_COOKIE_NAME
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise HTTPException(
            status_code=401,
            detail="No active session. Create a new project first."
        )
    session = get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=401,
            detail="Session expired. Create a new project."
        )
    return session
