"""Append-only run traces (JSONL)."""

import json
import uuid
from datetime import datetime, timezone

from .config import DATA_DIR, TRACE_FILE


def save_trace(result: dict, version: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    trace = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_version": version,
        **result,
    }
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")


def load_traces(version: str) -> list[dict]:
    """Traces of one policy version: older versions' failures are already handled."""
    if not TRACE_FILE.exists():
        return []
    traces = [json.loads(line) for line in TRACE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [t for t in traces if t["agent_version"] == version]
