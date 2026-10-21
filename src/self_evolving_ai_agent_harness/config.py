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

DATA_DIR = Path(os.getenv("HARNESS_DATA_DIR", "./self_evolution_data"))
TRACE_FILE = DATA_DIR / "traces.jsonl"
VERSION_FILE = DATA_DIR / "versions.json"


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
