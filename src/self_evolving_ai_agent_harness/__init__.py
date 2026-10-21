"""Self-Evolving Agent Harness: CrewAI agents on local Ollama models.

Usage:
    self-evolving-ai-agent-harness            serve traffic, evolve once, serve again
    self-evolving-ai-agent-harness rollback   reactivate the previous policy version
    self-evolving-ai-agent-harness reset      delete traces and versions
    self-evolving-ai-agent-harness selfcheck  offline check of the gates (no LLM)
"""

import shutil
import sys


def selfcheck() -> None:
    from .evaluation import quality_gate, regression_gate

    def bench(*oks):
        return {"score": sum(oks) / len(oks), "results": [{"query": i, "success": ok} for i, ok in enumerate(oks)]}

    v1 = bench(False, False, True)
    good = bench(True, True, True)
    naive = bench(True, True, False)  # fixes financial questions, breaks general search
    assert quality_gate(v1, good) and not regression_gate(v1, good)
    assert quality_gate(v1, naive) and regression_gate(v1, naive) == [2]
    assert not quality_gate(v1, v1)
    print("selfcheck OK")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "run"

    if command == "selfcheck":
        return selfcheck()

    from . import policy
    from .config import DATA_DIR, DECISION_MODEL, WORKER_MODEL

    if command == "reset":
        shutil.rmtree(DATA_DIR, ignore_errors=True)
        return print(f"Removed {DATA_DIR}")
    if command == "rollback":
        restored = policy.rollback()
        return print(f"Active version: {restored.version}" if restored else "Nothing to roll back to.")

    from .harness import evolve, serve_traffic

    print(f"SELF-EVOLVING AGENT HARNESS | worker={WORKER_MODEL} decision={DECISION_MODEL}")
    current = policy.active_policy()
    serve_traffic(current, "LIVE TRAFFIC")
    current = evolve(current)
    serve_traffic(current, "LIVE TRAFFIC AFTER EVOLUTION")
