"""Critic, benchmark and the two promotion gates."""

from .agents import run_executor
from .policy import AgentPolicy

# (question, tool a correct agent should use). Disjoint from the live traffic in harness.py,
# so a candidate must generalize, not memorize the failures it was built from.
BENCHMARK = [
    ("What profit did Snapchat make?", "financial_api"),
    ("How much revenue does Coca-Cola bring in?", "financial_api"),
    ("What is YouTube's parent company's net income?", "financial_api"),
    ("What are Amazon's total assets?", "financial_api"),
    ("What is Microsoft's earnings per share?", "financial_api"),
    ("Who founded Netflix?", "web_search"),
    ("What is Nvidia known for?", "web_search"),
]


def critique(query: str, expected_tool: str, calls: list[dict], answer: str) -> dict:
    """Grade the FIRST tool call: right tool and usable data. A retry after a failed call still counts as a
    failure, because it wastes a call and only works when the model happens to guess better."""
    first = calls[0] if calls else None
    if first is None:
        root_cause = "no_tool_call"
    elif first["tool"] != expected_tool:
        root_cause = f"wrong_tool:{first['tool']}"
    else:
        root_cause = first["failure"]  # None when the call returned real data
    return {
        "query": query,
        "expected_tool": expected_tool,
        "first_call": {k: first[k] for k in ("tool", "args", "output")} if first else None,
        # Full trajectory: a failed first call followed by a successful retry shows what the agent already knew.
        "trajectory": [f"{c['tool']}{c['args']} -> {c['failure'] or 'OK'}" for c in calls],
        "answer": answer,
        "success": root_cause is None,
        "root_cause": root_cause,
    }


def run_and_critique(query: str, expected_tool: str, policy: AgentPolicy) -> dict:
    calls, answer = run_executor(query, policy)
    return critique(query, expected_tool, calls, answer)


def run_benchmark(policy: AgentPolicy) -> dict:
    results = [run_and_critique(q, expected, policy) for q, expected in BENCHMARK]
    return {"score": sum(r["success"] for r in results) / len(results), "results": results}


def quality_gate(baseline: dict, candidate: dict) -> bool:
    """Candidate must strictly improve the overall score."""
    return candidate["score"] > baseline["score"]


def regression_gate(baseline: dict, candidate: dict) -> list[str]:
    """Questions that passed before and fail now. Empty list means PASS."""
    now_ok = {r["query"]: r["success"] for r in candidate["results"]}
    return [r["query"] for r in baseline["results"] if r["success"] and not now_ok[r["query"]]]
