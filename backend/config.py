"""Backend configuration settings"""

import os

# CORS origins allowed to access the API
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:4173",
]

# Upload directory for temporary files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Session configuration
SESSION_COOKIE_NAME = "scout_session"
SESSION_MAX_AGE = 3600  # 1 hour
