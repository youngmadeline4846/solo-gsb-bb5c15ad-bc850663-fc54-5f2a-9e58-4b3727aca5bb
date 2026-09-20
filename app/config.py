"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os

DEFAULT_DB_PATH = os.path.join("data", "blends.db")


def get_db_path() -> str:
    """SQLite database file path (env: DB_PATH)."""
    return os.environ.get("DB_PATH", DEFAULT_DB_PATH)


def get_host() -> str:
    """Bind host used by `python -m app.main` (env: HOST)."""
    return os.environ.get("HOST", "0.0.0.0")


def get_port() -> int:
    """Bind port used by `python -m app.main` (env: PORT)."""
    return int(os.environ.get("PORT", "8000"))
