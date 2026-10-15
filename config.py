"""Runtime settings for Kurios Kids, loaded from the environment."""

import os

from dotenv import load_dotenv

load_dotenv()


def get_api_key() -> str:
    """Read the Gemini key at call time so tests can override it."""
    return os.getenv("GEMINI_API_KEY", "").strip()


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").strip().lower() == "true"
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
    if origin.strip()
]

MAX_QUESTION_CHARS = 280
