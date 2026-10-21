"""Self-Evolving Agent Harness: CrewAI agents on local Ollama models.

Usage:
    self-evolving-ai-agent-harness            serve traffic, evolve once, serve again
    self-evolving-ai-agent-harness rollback   reactivate the previous policy version
    self-evolving-ai-agent-harness reset      delete traces and versions
    self-evolving-ai-agent-harness selfcheck  offline checks: gates, critic, matching, storage, lock (no LLM)
"""

import os
import shutil
import sys
import tempfile


def selfcheck() -> None:
    os.environ["HARNESS_DATA_DIR"] = tempfile.mkdtemp()  # never touch real run data
    from .evaluation import critique, quality_gate, regression_gate
    from .mcp_server import _key
    from .policy import AgentPolicy
    from .traces import load_traces, run_lock, save_trace

    # Gates on pass rates
    def bench(*rates):
        return {"score": sum(rates) / len(rates), "results": [{"query": i, "pass_rate": r} for i, r in enumerate(rates)]}

    v1, good, naive = bench(0, 0, 1), bench(1, 1, 1), bench(1, 1, 0)  # naive breaks general search
    assert quality_gate(v1, good) and not regression_gate(v1, good)
    assert quality_gate(v1, naive) and regression_gate(v1, naive) == [2]
    assert not quality_gate(v1, v1)
    assert not regression_gate(bench(1, 1, 1), bench(1, 1, 0.67))  # one flaky run in 3 is tolerated

    # Critic: wrong company, failed first call with recovery
    ok = {"tool": "financial_api", "args": {}, "failure": None}
    chase = critique("Chase net income?", "financial_api", [{**ok, "output": "CHASE PACKAGING CORP ..."}], "", "JPMorgan")
    assert chase["root_cause"] == "wrong_company"
    google = critique("Google revenue?", "financial_api",
                      [{**ok, "failure": "company_not_found", "output": "ERROR[company_not_found]"},
                       {**ok, "args": {"company": "Alphabet"}, "output": "Alphabet Inc. ..."}], "", "Alphabet")
    assert google["root_cause"] == "company_not_found" and google["recovery"]["args"] == {"company": "Alphabet"}

    # Company matching ignores punctuation and legal words
    assert _key("The Coca-Cola Company") == _key("COCA COLA CO") == "cocacola"

    # SQLite traces + run lock
    save_trace(google, "v1")
    assert load_traces("v1")[0]["root_cause"] == "company_not_found" and not load_traces("v2")
    with run_lock():
        try:
            with run_lock():
                raise AssertionError("second lock must fail")
        except SystemExit:
            pass
    assert AgentPolicy("v9", "x", examples=("a",)).describe() == "'x' +1 examples"
    print("selfcheck OK")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command not in {"run", "reset", "rollback", "selfcheck"}:
        sys.exit(__doc__)
    if command == "selfcheck":
        return selfcheck()

    from . import policy
    from .config import DATA_DIR, DECISION_MODEL, WORKER_MODEL
    from .traces import SESSION_ID, run_lock

    with run_lock():  # one run per data directory
        if command == "reset":
            for item in DATA_DIR.iterdir():
                if item.name != "harness.lock":
                    shutil.rmtree(item) if item.is_dir() else item.unlink()
            return print(f"Cleared {DATA_DIR}")
        if command == "rollback":
            restored = policy.rollback()
            return print(f"Active version: {restored.version}" if restored else "Nothing to roll back to.")

        from .harness import evolve, serve_traffic

        print(f"SELF-EVOLVING AGENT HARNESS | worker={WORKER_MODEL} decision={DECISION_MODEL} session={SESSION_ID}")
        current = policy.active_policy()
        serve_traffic(current, "LIVE TRAFFIC")
        current = evolve(current)
        serve_traffic(current, "LIVE TRAFFIC AFTER EVOLUTION")
