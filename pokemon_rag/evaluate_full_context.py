"""Evaluate the current corpus in full against a saved single-query baseline."""

import argparse
import hashlib
import json
from pathlib import Path

import requests

from pokemon_rag.build_index import DATA_DIR
from pokemon_rag.evaluate_planner import (
    OBJECTIVE, REVIEW_METRICS, append_jsonl, read_jsonl, summarize, write_json,
)
from pokemon_rag.planner import MODEL, OLLAMA_URL, PokemonPlanner, make_payload


def run(baseline, output):
    baseline, output = Path(baseline), Path(output)
    protocol = json.loads((baseline / "protocol.json").read_text(encoding="utf-8"))
    rows = [row for row in read_jsonl(baseline / "results.jsonl") if row["condition"] == "with_rag"]
    cases = {case["id"]: case for case in protocol["cases"]}
    expected = {f"{case}-with_rag-{i}" for case in cases for i in range(1, protocol["repetitions"] + 1)}
    if len(rows) != len(expected) or {row["trial_id"] for row in rows} != expected:
        raise ValueError("Single-query baseline is incomplete or contains duplicate trials")
    chunk_path = DATA_DIR / "chunks.json"
    chunks_hash = hashlib.sha256(chunk_path.read_bytes()).hexdigest()
    manifest = json.loads((DATA_DIR / "index_metadata.json").read_text(encoding="utf-8"))
    if chunks_hash != manifest.get("chunks_sha256"):
        raise ValueError("Chunks differ from index metadata")
    chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
    chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
    if len(chunks) != 27 or len(chunks_by_id) != len(chunks):
        raise ValueError("Expected the original 27 uniquely identified chunks")
    planner = PokemonPlanner()
    if protocol["model"] != MODEL or protocol["objective"] != OBJECTIVE:
        raise ValueError("Baseline model or objective differs from the current planner")
    for row in rows:
        for doc in row["documents"]:
            if any(doc.get(key) != value for key, value in chunks_by_id.get(doc["id"], {}).items()) or doc["id"] not in chunks_by_id:
                raise ValueError("Corpus differs from the baseline retrieval documents")
        case = cases[row["scenario_id"]]
        current = make_payload(case["state"], OBJECTIVE, chunks)
        saved = row["request"]
        if (current["messages"][0] != saved["messages"][0]
                or current["messages"][1]["content"].split("WALKTHROUGH CONTEXT:\n", 1)[0]
                != saved["messages"][1]["content"].split("WALKTHROUGH CONTEXT:\n", 1)[0]
                or {k: v for k, v in current.items() if k != "messages"}
                != {k: v for k, v in saved.items() if k != "messages"}):
            raise ValueError("Prompt, model, schema, state, or settings differ from baseline")

    base_url = OLLAMA_URL.rsplit("/", 1)[0]
    tags = requests.get(base_url + "/tags", timeout=10)
    tags.raise_for_status()
    model_info = next((m for m in tags.json()["models"] if m.get("name") == MODEL or m.get("model") == MODEL), None)
    if not model_info or model_info.get("digest") != (protocol.get("model_info") or {}).get("digest"):
        raise ValueError("Loaded model digest differs from baseline")
    output.mkdir(parents=True, exist_ok=False)

    # Probe the actual rendered prompt for every state without changing num_ctx.
    preflight = {}
    for case_id, case in cases.items():
        payload = make_payload(case["state"], OBJECTIVE, chunks)
        payload["options"]["num_predict"] = 1
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        response = response.json()
        running = requests.get(base_url + "/ps", timeout=10)
        running.raise_for_status()
        model = next(m for m in running.json()["models"] if m.get("name") == MODEL)
        count, capacity = response["prompt_eval_count"], model["context_length"]
        if count >= capacity:
            raise RuntimeError(f"{case_id} prompt has {count} tokens but context capacity is {capacity}")
        preflight[case_id] = {"prompt_tokens": count, "context_length": capacity,
                              "remaining_tokens": capacity - count}
        print(f"Preflight {case_id}: {count}/{capacity} tokens", flush=True)

    write_json(output / "protocol.json", {**protocol, "context_mode": "full",
        "baseline": str(baseline), "chunks_sha256": chunks_hash, "preflight": preflight,
        "note": "Full-context prompt preflight uses one output token; trials use the unchanged baseline settings."})
    for number, old in enumerate(rows, 1):
        case = cases[old["scenario_id"]]
        result = planner.generate(case["state"], OBJECTIVE, chunks)
        response = result["response"]
        input_tokens = response.get("prompt_eval_count")
        capacity = preflight[case["id"]]["context_length"]
        if input_tokens != preflight[case["id"]]["prompt_tokens"]:
            raise RuntimeError(f"{old['trial_id']} prompt token count changed; possible truncation")
        record = {"trial_id": old["trial_id"], "scenario_id": old["scenario_id"],
                  "condition": "with_rag", "repetition": old["repetition"], "documents": chunks,
                  **result, "prompt_eval_count": input_tokens, "eval_count": response.get("eval_count"),
                  "exceeds_context_with_output": input_tokens + (response.get("eval_count") or 0) > capacity}
        append_jsonl(output / "results.jsonl", record)
        append_jsonl(output / "review.jsonl", {"trial_id": old["trial_id"],
            **{metric: None for metric in REVIEW_METRICS}, "notes": ""})
        print(f"{number}/{len(rows)} {old['trial_id']}: schema_valid={record['schema_valid']}", flush=True)
    summary = summarize(read_jsonl(output / "results.jsonl"), read_jsonl(output / "review.jsonl"))
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.baseline, args.output)


if __name__ == "__main__":
    main()
