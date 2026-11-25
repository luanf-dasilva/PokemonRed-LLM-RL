"""Run controlled corpus, prompt-placement and state-representation variants."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import requests

from pokemon_rag.build_index import DATA_DIR
from pokemon_rag.evaluate_planner import OBJECTIVE, REVIEW_METRICS, append_jsonl, read_jsonl, summarize, write_json
from pokemon_rag.planner import MODEL, OLLAMA_URL, PokemonPlanner, make_payload
from pokemon_rag.retrieval_experiments import generation_one_verified, retrieve_context
from pokemon_rag.retriever import PokemonRetriever

MODES = ("top4", "top8", "multi", "state_last", "history")


def historical_state(state):
    """Separate inventory and known completed events without inventing history."""
    result = {k: v for k, v in state.items()
              if k not in {"has_starter", "has_oaks_parcel", "delivered_oaks_parcel"}}
    result["inventory"] = {"oaks_parcel": int(state["has_oaks_parcel"])}
    events = []
    if state["has_starter"]:
        events.append("Obtained starter Pokemon")
    if state["has_oaks_parcel"] or state["delivered_oaks_parcel"]:
        events.append("Obtained Oak's Parcel")
    if state["delivered_oaks_parcel"]:
        events.append("Delivered Oak's Parcel")
    result["completed_events"] = events
    return result


def run(baseline, expanded, output, modes=MODES):
    baseline, expanded, output = Path(baseline), Path(expanded), Path(output)
    if not modes or any(mode not in MODES for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError("Invalid or repeated experiment mode")
    protocol = json.loads((baseline / "protocol.json").read_text(encoding="utf-8"))
    base_rows = read_jsonl(baseline / "results.jsonl")
    cases = protocol["cases"]
    if protocol["model"] != MODEL or protocol["objective"] != OBJECTIVE:
        raise ValueError("Baseline model or objective differs")
    base_url = OLLAMA_URL.rsplit("/", 1)[0]
    response = requests.get(base_url + "/tags", timeout=10)
    response.raise_for_status()
    model_info = next((m for m in response.json()["models"] if m.get("name") == MODEL or m.get("model") == MODEL), None)
    if not model_info or model_info.get("digest") != (protocol.get("model_info") or {}).get("digest"):
        raise ValueError("Model digest differs from baseline")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")

    retriever = PokemonRetriever(expanded)
    if len(retriever.chunks) != 274:
        raise ValueError("Expected the 274-chunk expanded index")
    eligible = sum(generation_one_verified(chunk) for chunk in retriever.chunks)
    original = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
    original_hash = hashlib.sha256((DATA_DIR / "chunks.json").read_bytes()).hexdigest()
    if len(original) != 27:
        raise ValueError("Expected the original 27 chunks")
    # Verify the full corpus matches the prior full-context experiment too.
    full_protocol = json.loads((DATA_DIR / "experiments/qwen-full-context/protocol.json").read_text(encoding="utf-8"))
    if full_protocol.get("chunks_sha256") != original_hash:
        raise ValueError("Original corpus differs from full-context baseline")
    if full_protocol.get("model_info", {}).get("digest") != model_info.get("digest"):
        raise ValueError("Full-context baseline uses a different model")
    planner = PokemonPlanner(retriever=retriever)
    contexts, context_times = {}, {}
    for case in cases:
        contexts[case["id"]] = {}
        for mode in modes:
            start = time.perf_counter()
            docs = (retrieve_context(retriever, case["state"], OBJECTIVE, mode)
                    if mode in {"top4", "top8", "multi"} else original)
            if not docs:
                raise RuntimeError(f"No eligible documents for {case['id']} {mode}")
            contexts[case["id"]][mode] = docs
            context_times[f"{case['id']}-{mode}"] = time.perf_counter() - start

    # Confirm no accidental change to the baseline prompt/schema/settings.
    for case in cases:
        row = next(r for r in base_rows if r["condition"] == "with_rag" and r["scenario_id"] == case["id"])
        expected = make_payload(case["state"], OBJECTIVE, row["documents"])
        if expected != row["request"]:
            raise ValueError(f"Baseline request differs for {case['id']}")

    output.mkdir(parents=True)
    preflight = {}
    for case in cases:
        for mode in modes:
            state = historical_state(case["state"]) if mode == "history" else case["state"]
            placement = "state_last" if mode == "state_last" else "state_first"
            payload = make_payload(state, OBJECTIVE, contexts[case["id"]][mode], placement=placement)
            payload["options"]["num_predict"] = 1
            response = requests.post(OLLAMA_URL, json=payload, timeout=300)
            response.raise_for_status()
            response = response.json()
            ps = requests.get(base_url + "/ps", timeout=10)
            ps.raise_for_status()
            model = next(m for m in ps.json()["models"] if m.get("name") == MODEL)
            count, capacity = response["prompt_eval_count"], model["context_length"]
            if count >= capacity:
                raise RuntimeError(f"{case['id']} {mode} prompt has {count}/{capacity} tokens")
            preflight[f"{case['id']}-{mode}"] = {"prompt_tokens": count, "context_length": capacity}
            print(f"Preflight {case['id']} {mode}: {count}/{capacity}", flush=True)

    write_json(output / "protocol.json", {**protocol, "modes": list(modes), "model_info": model_info,
        "expanded_chunks_sha256": hashlib.sha256((expanded / "chunks.json").read_bytes()).hexdigest(),
        "expanded_index_sha256": hashlib.sha256((expanded / "pokemon.index").read_bytes()).hexdigest(),
        "original_chunks_sha256": original_hash, "expanded_chunk_count": len(retriever.chunks),
        "eligible_chunk_count": eligible, "filter": "Gen I Bulbapedia walkthroughs and source-grounded curated facts; mixed-scope articles excluded",
        "context_retrieval_seconds": context_times, "preflight": preflight,
        "baseline": str(baseline), "history_mapping": "Known held or delivered parcel implies obtained; absent parcel alone does not imply never obtained"})
    trials = []
    rng = random.Random(protocol["seed"])
    for repetition in range(1, protocol["repetitions"] + 1):
        order = list(cases)
        rng.shuffle(order)
        for case in order:
            variants = list(modes)
            rng.shuffle(variants)
            trials.extend((case, repetition, mode) for mode in variants)
    for number, (case, repetition, mode) in enumerate(trials, 1):
        state = historical_state(case["state"]) if mode == "history" else case["state"]
        placement = "state_last" if mode == "state_last" else "state_first"
        docs = contexts[case["id"]][mode]
        trial_id = f"{case['id']}-{mode}-{repetition}"
        try:
            result = planner.generate(state, OBJECTIVE, docs, placement=placement)
            response = result["response"]
            probe = preflight[f"{case['id']}-{mode}"]
            if response.get("prompt_eval_count") != probe["prompt_tokens"]:
                raise RuntimeError(f"Input token count differs from preflight: {trial_id}")
            goal = result["goal"] or {}
            record = {"trial_id": trial_id, "scenario_id": case["id"], "condition": "with_rag",
                "mode": mode, "repetition": repetition, "state": state, "documents": docs, **result,
                "prompt_eval_count": response.get("prompt_eval_count"), "eval_count": response.get("eval_count"),
                "exceeds_context_with_output": response["prompt_eval_count"] + (response.get("eval_count") or 0) > probe["context_length"],
                "completed_parcel_target": bool(case["state"]["delivered_oaks_parcel"]
                    and goal.get("goal_type") != "INSUFFICIENT_CONTEXT" and "parcel" in goal.get("target", "").casefold())}
        except (requests.RequestException, ValueError, KeyError, RuntimeError) as exc:
            append_jsonl(output / "errors.jsonl", {"trial_id": trial_id, "error": f"{type(exc).__name__}: {exc}"})
            raise
        append_jsonl(output / "results.jsonl", record)
        append_jsonl(output / "review.jsonl", {"trial_id": trial_id, **{metric: None for metric in REVIEW_METRICS}, "notes": ""})
        print(f"{number}/{len(trials)} {trial_id}: schema_valid={record['schema_valid']}", flush=True)
    all_rows = read_jsonl(output / "results.jsonl")
    reviews = read_jsonl(output / "review.jsonl")
    summary = {}
    for mode in modes:
        selected = [row for row in all_rows if row["mode"] == mode]
        summary[mode] = summarize(selected, reviews)["with_rag"]
        summary[mode]["completed_parcel_target"] = sum(row["completed_parcel_target"] for row in selected)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()
    run(args.baseline, args.expanded, args.output, args.modes)


if __name__ == "__main__":
    main()
