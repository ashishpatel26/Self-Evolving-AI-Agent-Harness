"""Versioned agent policy + the version store (accept / reject / rollback)."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .config import DATA_DIR, VERSION_FILE, WORKER_MODEL


@dataclass(frozen=True)
class AgentPolicy:
    """Everything evolution may change. Three knobs, all data, never code."""

    version: str
    # Knob 1: natural-language tool usage instruction in the executor's prompt.
    routing_instruction: str
    # Knob 2: few-shot examples of correct FIRST calls, mined from the agent's own successful retries.
    examples: tuple[str, ...] = field(default_factory=tuple)
    # Knob 3: which local model executes tasks.
    worker_model: str = WORKER_MODEL
    rationale: str = ""

    def describe(self) -> str:
        parts = [repr(self.routing_instruction)]
        if self.examples:
            parts.append(f"+{len(self.examples)} examples")
        if self.worker_model != WORKER_MODEL:
            parts.append(f"model={self.worker_model}")
        return " ".join(parts)


# Neutral baseline: no sabotage. Evolution must find the model's real weaknesses from traces.
BASELINE = AgentPolicy(
    version="v1",
    routing_instruction="Choose the tool that best matches the user's request.",
    rationale="baseline",
)

# Store format:
# {"active": "<version>", "versions": [{...policy, score}], "rejected": [{change, score, broke}]}


def _load() -> dict:
    if VERSION_FILE.exists():
        store = json.loads(VERSION_FILE.read_text(encoding="utf-8"))
        store.setdefault("rejected", [])
        return store
    return {"active": None, "versions": [], "rejected": []}


def _save(store: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_policy(entry: dict) -> AgentPolicy:
    return AgentPolicy(
        version=entry["version"],
        routing_instruction=entry["routing_instruction"],
        examples=tuple(entry.get("examples", ())),
        worker_model=entry.get("worker_model", WORKER_MODEL),
        rationale=entry.get("rationale", ""),
    )


def active_policy() -> AgentPolicy:
    store = _load()
    for entry in store["versions"]:
        if entry["version"] == store["active"]:
            return _to_policy(entry)
    promote(BASELINE, score=None)
    return BASELINE


def promote(policy: AgentPolicy, score: float | None) -> None:
    store = _load()
    store["versions"].append({**asdict(policy), "score": score, "timestamp": _now()})
    store["active"] = policy.version
    _save(store)


def record_rejection(policy: AgentPolicy, score: float, broke: list[str]) -> None:
    """Remember failed proposals so the Improver does not suggest them again."""
    store = _load()
    store["rejected"].append({"change": policy.describe(), "score": score, "broke": len(broke), "timestamp": _now()})
    _save(store)


def rejected_history(limit: int = 6) -> list[dict]:
    return _load()["rejected"][-limit:]


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
