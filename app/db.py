"""SQLite persistence for saved blending plans."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import get_db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL UNIQUE,
    request_hash    TEXT NOT NULL,
    request_json    TEXT NOT NULL,
    response_json   TEXT NOT NULL,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

_lock = threading.Lock()


@contextmanager
def _db():
    path = Path(get_db_path())
    if str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the database file and schema if they do not exist yet."""
    with _lock, _db() as conn:
        conn.executescript(_SCHEMA)


def find_by_key(key: str) -> Optional[Dict[str, Any]]:
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM plans WHERE idempotency_key = ?", (key,)
        ).fetchone()
    return dict(row) if row else None


def find_by_id(plan_id: int) -> Optional[Dict[str, Any]]:
    with _db() as conn:
        row = conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
    return dict(row) if row else None


def save_plan(
    key: str,
    request_hash: str,
    request_json: str,
    status: str,
    body: Dict[str, Any],
) -> Tuple[bool, int, Dict[str, Any], Optional[str]]:
    """Atomically persist a plan.

    ``body`` is the response payload without ``plan_id``; it is filled in once
    the row id is known.  Returns ``(created, plan_id, stored_body,
    existing_request_hash)``.  When the idempotency key already exists
    (concurrent insert), ``created`` is False and the stored row is returned.
    """
    body = dict(body)
    with _lock, _db() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO plans (idempotency_key, request_hash, request_json,"
                " response_json, status) VALUES (?, ?, ?, '', ?)",
                (key, request_hash, request_json, status),
            )
        except sqlite3.IntegrityError:
            row = conn.execute(
                "SELECT id, request_hash, response_json FROM plans"
                " WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            return False, row["id"], json.loads(row["response_json"]), row["request_hash"]
        plan_id = cur.lastrowid
        body["plan_id"] = plan_id
        conn.execute(
            "UPDATE plans SET response_json = ? WHERE id = ?",
            (json.dumps(body, ensure_ascii=False), plan_id),
        )
        return True, plan_id, body, None
