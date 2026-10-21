"""Critic, benchmark and the two promotion gates."""

import re
from statistics import mean, pstdev

from .agents import run_executor
from .config import BENCH_REPEATS, QUALITY_MARGIN, REGRESSION_TOLERANCE
from .policy import AgentPolicy
from .telemetry import span

# (question, tool a correct agent should use, company the answer must be about).
# Disjoint from the live traffic in harness.py, so a candidate must generalize, not memorize.
BENCHMARK = [
    ("What profit did Snapchat make?", "financial_api", "Snap"),
    ("How much revenue does Coca-Cola bring in?", "financial_api", "Coca Cola Co"),
    ("What is YouTube's parent company's net income?", "financial_api", "Alphabet"),
    ("What are Amazon's total assets?", "financial_api", "Amazon"),
    ("What is Microsoft's earnings per share?", "financial_api", "Microsoft"),
    # Held-out brand aliases that v1 fails on the first call (registrant is WALT DISNEY CO / Alphabet Inc.).
    # Without unseen examples of the failure class, v1 scores 100% and no fix can pass the quality gate.
    ("How much revenue did Disney make?", "financial_api", "Walt Disney"),
    ("What are Gmail maker's total assets?", "financial_api", "Alphabet"),
    ("Who founded Netflix?", "web_search", None),
    ("What is Nvidia known for?", "web_search", None),
]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def critique(query: str, expected_tool: str, calls: list[dict], answer: str, expected_company: str | None = None) -> dict:
    """Grade the FIRST tool call: right tool, usable data, right company. A retry after a failed call still
    counts as a failure, because it wastes a call and only works when the model happens to guess better."""
    first = calls[0] if calls else None
    if first is None:
        root_cause = "no_tool_call"
    elif first["tool"] != expected_tool:
        root_cause = f"wrong_tool:{first['tool']}"
    elif first["failure"]:
        root_cause = first["failure"]
    elif expected_company and _norm(expected_company) not in _norm(first["output"]):
        root_cause = "wrong_company"  # e.g. 'Chase' silently resolved to CHASE PACKAGING CORP
    else:
        root_cause = None
    # The first later call that worked: evidence of what the agent already knew (mined into few-shot examples).
    recovery = next((c for c in calls[1:] if c["tool"] == expected_tool and not c["failure"]), None)
    return {
        "query": query,
        "expected_tool": expected_tool,
        "first_call": {k: first[k] for k in ("tool", "args", "output")} if first else None,
        "trajectory": [f"{c['tool']}{c['args']} -> {c['failure'] or 'OK'}" for c in calls],
        "recovery": {"tool": recovery["tool"], "args": recovery["args"]} if recovery and root_cause else None,
        "answer": answer,
        "success": root_cause is None,
        "root_cause": root_cause,
    }


def run_and_critique(query: str, expected_tool: str, policy: AgentPolicy, expected_company: str | None = None) -> dict:
    with span("harness.task", query=query, policy=policy.version) as s:
        calls, answer = run_executor(query, policy)
        result = critique(query, expected_tool, calls, answer, expected_company)
        s.set_attribute("success", result["success"])
        s.set_attribute("root_cause", result["root_cause"] or "")
        s.set_attribute("tool_calls", len(calls))
        return result


def run_benchmark(policy: AgentPolicy, repeats: int = BENCH_REPEATS) -> dict:
    """Run every question `repeats` times. Per question: pass_rate. Overall: mean score and its spread."""
    with span("harness.benchmark", policy=policy.version, repeats=repeats) as s:
        runs = [[run_and_critique(q, tool, policy, company)["success"] for q, tool, company in BENCHMARK]
                for _ in range(repeats)]
        results = [{"query": q, "pass_rate": mean(run[i] for run in runs)} for i, (q, _, _) in enumerate(BENCHMARK)]
        per_run_scores = [mean(run) for run in runs]
        score, spread = mean(per_run_scores), pstdev(per_run_scores)
        s.set_attribute("score", score)
        return {"score": score, "spread": spread, "repeats": repeats, "results": results}


def quality_gate(baseline: dict, candidate: dict) -> bool:
    """Candidate's mean score must beat the active policy by more than QUALITY_MARGIN."""
    return candidate["score"] > baseline["score"] + QUALITY_MARGIN


def regression_gate(baseline: dict, candidate: dict) -> list[str]:
    """Questions whose pass rate dropped by more than REGRESSION_TOLERANCE. Empty list means PASS."""
    now = {r["query"]: r["pass_rate"] for r in candidate["results"]}
    return [r["query"] for r in baseline["results"] if r["pass_rate"] - now[r["query"]] > REGRESSION_TOLERANCE]
