"""Run paired, repeatable planner trials and prepare human review data."""

import argparse
import hashlib
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from pokemon_rag.planner import MODEL, OLLAMA_URL, PokemonPlanner

CASES = Path(__file__).with_name("planner_cases.json")
OBJECTIVE = "Obtain the Boulder Badge"
REVIEW_METRICS = ("factual_accuracy", "prerequisite_correctness", "goal_validity", "grounding_accuracy")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(results, reviews):
    review_by_id = {row["trial_id"]: row for row in reviews}
    summary = {}
    for mode in ("without_rag", "with_rag"):
        rows = [row for row in results if row["condition"] == mode]
        stats = {"trials": len(rows), "schema_valid": sum(r.get("schema_valid") is True for r in rows)}
        stats["schema_valid_rate"] = stats["schema_valid"] / len(rows) if rows else None
        stats["evidence_present"] = sum(r.get("evidence_present") is True for r in rows)
        stats["valid_cited_ids_with_evidence"] = sum(
            r.get("citation_ids_valid") is True and r.get("evidence_present") is True for r in rows
        )
        stats["invalid_citation_ids"] = sum(r.get("citation_ids_valid") is False for r in rows)
        stats["insufficient_context"] = sum(
            (r.get("goal") or {}).get("goal_type") == "INSUFFICIENT_CONTEXT" for r in rows
        )
        for field, source in (("mean_latency_seconds", "latency_seconds"),
                              ("mean_prompt_tokens", "prompt_eval_count"),
                              ("mean_output_tokens", "eval_count")):
            values = [r.get(source) for r in rows if isinstance(r.get(source), (int, float))]
            stats[field] = statistics.mean(values) if values else None
        for metric in REVIEW_METRICS:
            labels = [review_by_id.get(row["trial_id"], {}).get(metric) for row in rows]
            labels = [label for label in labels if isinstance(label, bool)]
            stats[metric] = {"reviewed": len(labels), "pass_rate": sum(labels) / len(labels) if labels else None}
        # Exact equality is descriptive only; semantic equivalence needs human review.
        grouped = {}
        for row in rows:
            goal = row.get("goal")
            key = (goal["goal_type"], goal["target"].strip().casefold(), goal["target_value"]) if goal else None
            grouped.setdefault(row["scenario_id"], []).append(key)
        stats["exact_goal_consistency"] = {
            scenario: (None if any(value is None for value in values) else len(set(values)) == 1)
            for scenario, values in grouped.items()
        }
        summary[mode] = stats
    return summary


def run(output, repetitions=5, seed=0):
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    planner = PokemonPlanner()
    context = {}
    context_latency = {}
    for case in cases:
        start = time.perf_counter()
        context[case["id"]] = planner.get_context(case["state"], OBJECTIVE)
        context_latency[case["id"]] = time.perf_counter() - start

    # Shuffle condition order within each pair to reduce order effects.
    rng = random.Random(seed)
    trials = []
    for repetition in range(1, repetitions + 1):
        case_order = list(cases)
        rng.shuffle(case_order)
        for case in case_order:
            conditions = ["without_rag", "with_rag"]
            rng.shuffle(conditions)
            trials.extend((case, repetition, condition) for condition in conditions)

    tags_url = OLLAMA_URL.rsplit("/", 1)[0] + "/tags"
    response = requests.get(tags_url, timeout=10)
    response.raise_for_status()
    tags = response.json().get("models", [])
    model_info = next((m for m in tags if m.get("name") == MODEL or m.get("model") == MODEL), None)
    write_json(output / "protocol.json", {
        "model": MODEL, "model_info": model_info, "objective": OBJECTIVE,
        "repetitions": repetitions, "seed": seed, "cases": cases,
        "case_file_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
        "context_retrieval_seconds": context_latency,
        "note": "Human judgments are unset; citation ID validity does not establish factual grounding.",
    })
    results_path = output / "results.jsonl"
    review_path = output / "review.jsonl"
    for case, repetition, condition in trials:
        trial_id = f"{case['id']}-{condition}-{repetition}"
        documents = context[case["id"]] if condition == "with_rag" else []
        record = {"trial_id": trial_id, "scenario_id": case["id"], "condition": condition,
                  "repetition": repetition, "documents": documents}
        try:
            result = planner.generate(case["state"], OBJECTIVE, documents)
            record.update(result)
            record["prompt_eval_count"] = result["response"].get("prompt_eval_count")
            record["eval_count"] = result["response"].get("eval_count")
        except (requests.RequestException, ValueError, KeyError) as exc:
            record.update({"schema_valid": False, "error": f"{type(exc).__name__}: {exc}"})
            append_jsonl(results_path, record)
            raise RuntimeError(f"Trial {trial_id} failed; partial results saved to {output}") from exc
        append_jsonl(results_path, record)
        append_jsonl(review_path, {
            "trial_id": trial_id, **{metric: None for metric in REVIEW_METRICS}, "notes": ""
        })
        print(f"{len(read_jsonl(results_path))}/{len(trials)} {trial_id}: schema_valid={record['schema_valid']}", flush=True)
    summary = summarize(read_jsonl(results_path), read_jsonl(review_path))
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory for experiment artifacts")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summarize", type=Path, help="Recompute summary after human review")
    args = parser.parse_args()
    if args.summarize:
        summary = summarize(read_jsonl(args.summarize / "results.jsonl"), read_jsonl(args.summarize / "review.jsonl"))
        write_json(args.summarize / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    else:
        output = args.output or Path("pokemon_rag/data/experiments") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        print(f"Saving experiment to {output}", flush=True)
        run(output, args.repetitions, args.seed)


if __name__ == "__main__":
    main()
