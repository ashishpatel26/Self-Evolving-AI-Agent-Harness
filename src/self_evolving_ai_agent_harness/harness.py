"""The self-evolution loop.

    live traffic -> traces -> failure clusters -> Diagnoser -> Improver -> candidates
    -> Decider picks one -> quality gate + regression gate -> ACCEPT (promote) / REJECT (try next)

The LLMs only propose and choose. The harness runs the benchmark and makes the final call.
"""

from collections import Counter

from . import agents
from .config import DECISION_MODEL, WORKER_MODEL
from .evaluation import quality_gate, regression_gate, run_and_critique, run_benchmark
from .policy import AgentPolicy, next_version, promote
from .traces import load_traces, save_trace

# Simulated production traffic: (question, correct tool)
LIVE_TRAFFIC = [
    ("How much money did Google make last year?", "financial_api"),
    ("What did Facebook earn in profit?", "financial_api"),
    ("What is Apple's EPS?", "financial_api"),
    ("How much revenue did Tesla make?", "financial_api"),
    ("Who is the CEO of Apple?", "web_search"),
]

# Always offered to the Decider: the tempting over-correction the regression gate must catch.
NAIVE_FIX = "Always use financial_api for every question."

MAX_ATTEMPTS = 2


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def serve_traffic(policy: AgentPolicy, title: str) -> None:
    banner(f"{title}  [policy {policy.version}: {policy.routing_instruction!r}]")
    for query, expected in LIVE_TRAFFIC:
        result = run_and_critique(query, expected, policy)
        save_trace(result, policy.version)
        first = result["first_call"] or {}
        status = "OK  " if result["success"] else f"FAIL[{result['root_cause']}]"
        print(f"{status:28} {query:45} first_call={first.get('tool')}{first.get('args')}")
        print(f"{'':28} -> {result['answer'][:110]!r}")


def print_comparison(current: AgentPolicy, base: dict, cand: dict) -> None:
    print(f"  {'benchmark question':45} {current.version:>8} {'candidate':>10}")
    for b, c in zip(base["results"], cand["results"]):
        print(f"  {b['query']:45} {'OK' if b['success'] else 'FAIL':>8} {'OK' if c['success'] else 'FAIL':>10}")
    print(f"  {'SCORE':45} {base['score']:>8.0%} {cand['score']:>10.0%}")


def evolve(policy: AgentPolicy) -> AgentPolicy:
    banner("SELF-EVOLUTION CYCLE")

    # 1. Failure clustering
    failures = [t for t in load_traces(policy.version) if not t["success"]]
    if not failures:
        print("No failures for the active version. Nothing to evolve.")
        return policy
    clusters = Counter(f["root_cause"] for f in failures)
    for cause, count in clusters.most_common():
        print(f"cluster {cause}: {count} runs")
    pattern, _ = clusters.most_common(1)[0]
    cluster = [f for f in failures if f["root_cause"] == pattern]

    # 2. Diagnose
    print(f"\n[Diagnoser · {WORKER_MODEL}]")
    diagnosis = agents.diagnose(cluster)
    print(diagnosis)

    # 3. Propose
    print(f"\n[Improver · {WORKER_MODEL}]")
    options = agents.propose_improvements(diagnosis, policy) + [NAIVE_FIX]
    for i, option in enumerate(options):
        print(f"  {'ABCDEFGH'[i]}) {option}")

    # 4. Decide + gate. Rejected options are dropped and the Decider chooses again.
    baseline = run_benchmark(policy)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        pick = agents.decide(pattern, policy.routing_instruction, options)
        chosen = options.pop(pick)
        print(f"\n[Decider · {DECISION_MODEL}] attempt {attempt}: {chosen!r}")

        candidate = AgentPolicy(next_version(), chosen, rationale=diagnosis[:300])
        result = run_benchmark(candidate)
        print_comparison(policy, baseline, result)

        quality_ok = quality_gate(baseline, result)
        regressions = regression_gate(baseline, result)
        print(f"  Quality gate   : {'PASS' if quality_ok else 'FAIL'}")
        print(f"  Regression gate: {'PASS' if not regressions else f'FAIL (broke {regressions})'}")

        if quality_ok and not regressions:
            promote(candidate, result["score"])
            print(f"\nUPDATE ACCEPTED -> {candidate.version} is now active")
            return candidate
        print("  REJECTED")
        if not options:
            break

    print(f"\nNo candidate passed the gates. {policy.version} stays active.")
    return policy
