"""Backend configuration settings"""

import os

# CORS origins allowed to access the API
# In production (Docker), frontend is served from same origin so CORS isn't needed.
# These are for local development where frontend runs on a separate port.
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:4173,http://127.0.0.1:5173,http://127.0.0.1:4173").split(",")

# Upload directory for temporary files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Session configuration
SESSION_COOKIE_NAME = "scout_session"
SESSION_MAX_AGE = 3600  # 1 hour
