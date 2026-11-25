"""Export experiment outcomes without copying retrieved source passages."""

import argparse
import json
from pathlib import Path

from pokemon_rag.evaluate_planner import read_jsonl


def export(source, destination):
    rows = read_jsonl(Path(source) / "results.jsonl")
    if not rows:
        raise ValueError("No experiment results to export")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = ("trial_id", "scenario_id", "condition", "mode", "repetition", "goal", "schema_valid",
            "citation_ids_valid", "evidence_present", "latency_seconds", "prompt_eval_count", "eval_count",
            "completed_parcel_target", "exceeds_context_with_output")
    with destination.open("w", encoding="utf-8") as out:
        for row in rows:
            item = {key: row[key] for key in keys if key in row}
            item["context_ids"] = [doc["id"] for doc in row.get("documents", [])]
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"Exported {export(args.source, args.destination)} trial outcomes")
