"""Run data: SQLite trace store + a run lock so two harness runs never share one data directory."""

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import DATA_DIR, LOCK_FILE, TRACE_DB

SESSION_ID = uuid.uuid4().hex[:12]  # one per harness process
LOCK_STALE_SECONDS = 3 * 3600


def _db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(TRACE_DB)
    db.execute(
        """CREATE TABLE IF NOT EXISTS traces (
               run_id TEXT PRIMARY KEY, session_id TEXT, timestamp TEXT,
               agent_version TEXT, success INTEGER, root_cause TEXT, data TEXT)"""
    )
    db.execute("CREATE INDEX IF NOT EXISTS traces_version ON traces(agent_version)")
    return db


def save_trace(result: dict, version: str) -> None:
    with _db() as db:  # commits on success
        db.execute(
            "INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, SESSION_ID, datetime.now(timezone.utc).isoformat(), version,
             int(result["success"]), result["root_cause"], json.dumps(result)),
        )


def load_traces(version: str) -> list[dict]:
    """Traces of one policy version: older versions' failures are already handled."""
    with _db() as db:
        rows = db.execute(
            "SELECT data FROM traces WHERE agent_version = ? ORDER BY timestamp", (version,)
        ).fetchall()
    return [json.loads(data) for (data,) in rows]


@contextmanager
def run_lock():
    """Exclusive lock on DATA_DIR. A lock older than LOCK_STALE_SECONDS is treated as a crashed run."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists() and time.time() - LOCK_FILE.stat().st_mtime > LOCK_STALE_SECONDS:
        LOCK_FILE.unlink()
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        owner = LOCK_FILE.read_text(encoding="utf-8")
        raise SystemExit(
            f"Another harness run holds {LOCK_FILE} ({owner}).\n"
            "Wait for it, or use a separate HARNESS_DATA_DIR for this run."
        )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()} session={SESSION_ID}")
    try:
        yield
    finally:
        LOCK_FILE.unlink(missing_ok=True)
