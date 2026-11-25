"""Verify and compare two paired planner runs that differ only by prompt."""

import argparse
import json
from pathlib import Path

from pokemon_rag.evaluate_planner import read_jsonl, summarize


def compare(original, revised):
    original, revised = Path(original), Path(revised)
    protocols = [json.loads((path / "protocol.json").read_text(encoding="utf-8"))
                 for path in (original, revised)]
    for field in ("model", "objective", "repetitions", "seed", "cases", "case_file_sha256"):
        if protocols[0].get(field) != protocols[1].get(field):
            raise ValueError(f"Experiments differ in {field}")
    digests = [(protocol.get("model_info") or {}).get("digest") for protocol in protocols]
    if not digests[0] or digests[0] != digests[1]:
        raise ValueError("Model digests are missing or do not match")

    result_lists = [read_jsonl(path / "results.jsonl") for path in (original, revised)]
    expected = {
        f"{case['id']}-{condition}-{repetition}"
        for case in protocols[0]["cases"]
        for condition in ("without_rag", "with_rag")
        for repetition in range(1, protocols[0]["repetitions"] + 1)
    }
    results = [{row["trial_id"]: row for row in rows} for rows in result_lists]
    if any(len(rows) != len(expected) or set(indexed) != expected
           for rows, indexed in zip(result_lists, results)):
        raise ValueError("Experiments are incomplete or contain duplicate trials")

    old_prompts, new_prompts = set(), set()
    for trial_id in expected:
        first, second = (indexed[trial_id] for indexed in results)
        if first["documents"] != second["documents"]:
            raise ValueError(f"Retrieved documents differ for {trial_id}")
        old_request, new_request = first["request"], second["request"]
        old_messages, new_messages = old_request["messages"], new_request["messages"]
        if (len(old_messages) != 2 or len(new_messages) != 2
                or old_messages[0]["role"] != "system" or new_messages[0]["role"] != "system"
                or old_messages[1:] != new_messages[1:]
                or {k: v for k, v in old_request.items() if k != "messages"}
                != {k: v for k, v in new_request.items() if k != "messages"}):
            raise ValueError(f"Generation settings or state prompt differ for {trial_id}")
        old_prompts.add(old_messages[0]["content"])
        new_prompts.add(new_messages[0]["content"])
    if len(old_prompts) != 1 or len(new_prompts) != 1 or old_prompts == new_prompts:
        raise ValueError("Each run must use one distinct system prompt")
    return {
        "comparison": "Prompt-only ablation; scenarios, model digest, context and other request settings match",
        "original": summarize(result_lists[0], read_jsonl(original / "review.jsonl")),
        "revised": summarize(result_lists[1], read_jsonl(revised / "review.jsonl")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("revised", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(compare(args.original, args.revised), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
