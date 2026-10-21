"""Versioned agent policy + the version store (accept / rollback)."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .config import DATA_DIR, VERSION_FILE


@dataclass(frozen=True)
class AgentPolicy:
    version: str
    # The only thing evolution may change: a natural-language instruction in the executor's prompt.
    routing_instruction: str
    rationale: str = ""


# Neutral baseline: no sabotage. Evolution must find the model's real weaknesses from traces.
BASELINE = AgentPolicy(
    version="v1",
    routing_instruction="Choose the tool that best matches the user's request.",
    rationale="baseline",
)

# Store format: {"active": "<version>", "versions": [{...policy, "score", "timestamp"}]}


def _load() -> dict:
    if VERSION_FILE.exists():
        return json.loads(VERSION_FILE.read_text(encoding="utf-8"))
    return {"active": None, "versions": []}


def _save(store: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _to_policy(entry: dict) -> AgentPolicy:
    return AgentPolicy(entry["version"], entry["routing_instruction"], entry.get("rationale", ""))


def active_policy() -> AgentPolicy:
    store = _load()
    for entry in store["versions"]:
        if entry["version"] == store["active"]:
            return _to_policy(entry)
    promote(BASELINE, score=None)
    return BASELINE


def promote(policy: AgentPolicy, score: float | None) -> None:
    store = _load()
    store["versions"].append({**asdict(policy), "score": score, "timestamp": datetime.now(timezone.utc).isoformat()})
    store["active"] = policy.version
    _save(store)


def next_version() -> str:
    return f"v{len(_load()['versions']) + 1}"


def rollback() -> AgentPolicy | None:
    """Reactivate the version promoted before the active one."""
    store = _load()
    ids = [v["version"] for v in store["versions"]]
    if store["active"] not in ids or ids.index(store["active"]) == 0:
        return None
    previous = store["versions"][ids.index(store["active"]) - 1]
    store["active"] = previous["version"]
    _save(store)
    return _to_policy(previous)
