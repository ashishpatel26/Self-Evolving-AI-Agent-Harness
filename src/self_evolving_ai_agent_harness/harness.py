"""The self-evolution loop.

    live traffic -> traces -> failure clusters -> Diagnoser -> candidates from 3 knobs
        (Improver instructions · few-shot examples mined from retries · worker model)
    -> Decider picks one -> quality gate + regression gate -> ACCEPT (promote) / REJECT (remember, try next)

The LLMs only propose and choose. The harness runs the benchmark and makes the final call.
"""

from collections import Counter
from dataclasses import replace

from . import agents
from .config import CANDIDATE_MODELS, DECISION_MODEL, WORKER_MODEL
from .evaluation import quality_gate, regression_gate, run_and_critique, run_benchmark
from .policy import AgentPolicy, next_version, promote, record_rejection, rejected_history
from .telemetry import span
from .traces import load_traces, save_trace

# Simulated production traffic: (question, correct tool, company the answer must be about)
LIVE_TRAFFIC = [
    ("How much money did Google make last year?", "financial_api", "Alphabet"),
    ("What did Facebook earn in profit?", "financial_api", "Meta Platforms"),
    ("What is Apple's EPS?", "financial_api", "Apple"),
    ("How much revenue did Tesla make?", "financial_api", "Tesla"),
    ("Who is the CEO of Apple?", "web_search", None),
]

# Always offered to the Decider: the tempting over-correction the regression gate must catch.
NAIVE_FIX = "Always use financial_api for every question."

MAX_ATTEMPTS = 3
LETTERS = "ABCDEFGH"


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def serve_traffic(policy: AgentPolicy, title: str) -> None:
    banner(f"{title}  [policy {policy.version}: {policy.describe()}]")
    for query, expected, company in LIVE_TRAFFIC:
        result = run_and_critique(query, expected, policy, company)
        save_trace(result, policy.version)
        first = result["first_call"] or {}
        status = "OK  " if result["success"] else f"FAIL[{result['root_cause']}]"
        print(f"{status:28} {query:45} first_call={first.get('tool')}{first.get('args')}")
        print(f"{'':28} -> {result['answer'][:110]!r}")


def print_comparison(current: AgentPolicy, base: dict, cand: dict) -> None:
    print(f"  {'benchmark question (pass rate)':47} {current.version:>8} {'candidate':>10}")
    for b, c in zip(base["results"], cand["results"]):
        print(f"  {b['query']:47} {b['pass_rate']:>8.0%} {c['pass_rate']:>10.0%}")
    spread = lambda r: f" ±{r['spread']:.0%}" if r["repeats"] > 1 else ""  # noqa: E731
    print(f"  {'SCORE (mean of ' + str(base['repeats']) + ' runs)':47} "
          f"{base['score']:>5.0%}{spread(base):>4} {cand['score']:>7.0%}{spread(cand)}")


def mine_examples(failures: list[dict]) -> tuple[str, ...]:
    """Turn 'first call failed, retry worked' into few-shot examples of the correct FIRST call."""
    examples = {}
    for f in failures:
        if f.get("recovery"):
            args = ", ".join(f"{k}={v!r}" for k, v in f["recovery"]["args"].items())
            examples[f["query"]] = f"{f['query']!r} -> {f['recovery']['tool']}({args})"
    return tuple(examples.values())


def build_candidates(policy: AgentPolicy, diagnosis: str, failures: list[dict]) -> list[tuple[str, AgentPolicy]]:
    """(description for the Decider, candidate policy) for every knob evolution can turn."""
    candidates: list[tuple[str, AgentPolicy]] = []

    # Knob 1: instruction rewrites from the Improver (steered by recoveries + rejected history)
    recoveries = [
        f"{f['query']!r}: first call {f['first_call']['args'] if f['first_call'] else None} failed, "
        f"retry {f['recovery']['args']} worked" for f in failures if f.get("recovery")
    ]
    for text in agents.propose_improvements(diagnosis, policy, recoveries, rejected_history()):
        candidates.append((text, replace(policy, routing_instruction=text)))

    # Knob 2: few-shot examples mined from the agent's own successful retries (self-distillation)
    examples = mine_examples(failures)
    if examples and set(examples) != set(policy.examples):
        merged = tuple(dict.fromkeys(policy.examples + examples))
        candidates.append((
            f"Keep the current instruction and add examples of correct first calls: {'; '.join(examples)}",
            replace(policy, examples=merged),
        ))

    # Knob 3: model choice (only models you list in CANDIDATE_MODELS)
    for model in CANDIDATE_MODELS:
        if model != policy.worker_model:
            candidates.append((f"Keep the policy but execute tasks with model {model}", replace(policy, worker_model=model)))

    candidates.append((NAIVE_FIX, replace(policy, routing_instruction=NAIVE_FIX)))
    return candidates


def evolve(policy: AgentPolicy) -> AgentPolicy:
    banner("SELF-EVOLUTION CYCLE")
    with span("harness.evolve", policy=policy.version) as cycle:
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
        cycle.set_attribute("pattern", pattern)

        # 2. Diagnose
        print(f"\n[Diagnoser · {WORKER_MODEL}]")
        diagnosis = agents.diagnose(cluster)
        print(diagnosis)

        # 3. Propose (all knobs)
        print(f"\n[Improver · {WORKER_MODEL}] + mined examples + model choice")
        candidates = build_candidates(policy, diagnosis, cluster)
        for letter, (text, _) in zip(LETTERS, candidates):
            print(f"  {letter}) {text}")

        # 4. Decide + gate. Rejected candidates are remembered, dropped, and the Decider chooses again.
        baseline = run_benchmark(policy)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            pick = agents.decide(pattern, policy.describe(), [text for text, _ in candidates])
            text, candidate = candidates.pop(pick)
            candidate = replace(candidate, version=next_version(), rationale=diagnosis[:300])
            print(f"\n[Decider · {DECISION_MODEL}] attempt {attempt}: {text!r}")

            with span("harness.gate", candidate=candidate.describe()) as gate:
                result = run_benchmark(candidate)
                print_comparison(policy, baseline, result)
                quality_ok = quality_gate(baseline, result)
                regressions = regression_gate(baseline, result)
                gate.set_attribute("accepted", quality_ok and not regressions)
            print(f"  Quality gate   : {'PASS' if quality_ok else 'FAIL'}")
            print(f"  Regression gate: {'PASS' if not regressions else f'FAIL (broke {regressions})'}")

            if quality_ok and not regressions:
                promote(candidate, result["score"])
                print(f"\nUPDATE ACCEPTED -> {candidate.version} is now active")
                return candidate
            record_rejection(candidate, result["score"], regressions)
            print("  REJECTED (remembered for the next cycle)")
            if not candidates:
                break

        print(f"\nNo candidate passed the gates. {policy.version} stays active.")
        return policy
