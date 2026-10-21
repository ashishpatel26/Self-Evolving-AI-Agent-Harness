"""Settings and the two local Ollama models used by the agents."""

import os
from pathlib import Path

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")  # no interactive "share trace?" prompt
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from crewai import LLM  # noqa: E402  (env vars must be set before import)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Worker model: executes tasks, diagnoses failures, drafts improvements.
WORKER_MODEL = os.getenv("WORKER_MODEL", "qwen3.5-4b-16k:latest")
# Decision model: fine-tuned to pick exactly one option letter from a decision task.
DECISION_MODEL = os.getenv("DECISION_MODEL", "tev1:latest")

# Extra worker models the harness may try as a "model choice" candidate (comma-separated, must support tools).
CANDIDATE_MODELS = [m.strip() for m in os.getenv("CANDIDATE_MODELS", "").split(",") if m.strip()]

DATA_DIR = Path(os.getenv("HARNESS_DATA_DIR", "./self_evolution_data"))
TRACE_DB = DATA_DIR / "traces.db"
VERSION_FILE = DATA_DIR / "versions.json"
LOCK_FILE = DATA_DIR / "harness.lock"

# Benchmark statistics: run each question N times to absorb LLM nondeterminism.
BENCH_REPEATS = int(os.getenv("BENCH_REPEATS", "1"))
# Regression = a question's pass rate drops by more than this (with N=1 any 1 -> 0 drop counts).
REGRESSION_TOLERANCE = float(os.getenv("REGRESSION_TOLERANCE", "0.34"))
# Quality = candidate mean score beats the active one by more than this margin.
QUALITY_MARGIN = float(os.getenv("QUALITY_MARGIN", "0.0"))


def ollama_llm(model: str, max_tokens: int = 1500) -> LLM:
    return LLM(
        model=f"ollama/{model}",
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        max_tokens=max_tokens,
        timeout=300,
        # Qwen 3.5 models think by default and leave `content` empty.
        # CrewAI only forwards reasoning_effort for OpenAI models, so send it raw.
        additional_params={"extra_body": {"reasoning_effort": "none"}},
    )
