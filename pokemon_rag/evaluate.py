"""Evaluate walkthrough retrieval using fixed, human-labelled queries."""

import json
from pathlib import Path

from pokemon_rag.retriever import PokemonRetriever

CASES_PATH = Path(__file__).with_name("evaluation_cases.json")
STATE = {
    "location": "Viridian City",
    "party_levels": [7],
    "hp_fraction": 0.8,
    "has_oaks_parcel": False,
    "delivered_oaks_parcel": False,
    "badges": 0,
}
OBJECTIVE = "Obtain the Boulder Badge"
STATE_RELEVANT = [
    {"part": 2, "section": "Viridian City > Poké Mart > Special Delivery"},
    {"part": 2, "section": "Pallet Town > Oak's Lab"},
    {"part": 2, "section": "Viridian City", "text_contains": "Oak's Parcel delivered"},
]


def is_relevant(document, labels):
    return any(
        document["part"] == label["part"]
        and document["section"] == label["section"]
        and label.get("text_contains", "").casefold() in document["text"].casefold()
        for label in labels
    )


def evaluate(retriever, cases, k=4):
    if not cases:
        raise ValueError("Evaluation requires at least one query")
    hits = 0
    reciprocal_ranks = []
    for case in cases:
        documents = retriever.retrieve(case["query"], k=k)
        rank = next((i for i, doc in enumerate(documents, 1) if is_relevant(doc, case["relevant"])), None)
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        print(f"{case['id']}: {'rank ' + str(rank) if rank else 'no hit'}")
        for doc in documents:
            print(f"  {doc['score']:.3f} part {doc['part']} {doc['section']} [{doc['id']}]")
    return hits / len(cases), sum(reciprocal_ranks) / len(cases)


def state_query(state, objective):
    return (
        f"Current location: {state['location']}. Objective: {objective}. "
        f"Current game state: {json.dumps(state)}. "
        "What prerequisite or next step is required?"
    )


def main():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    retriever = PokemonRetriever()
    hit_rate, mrr = evaluate(retriever, cases)
    print(f"Hit Rate@4: {hit_rate:.3f}; MRR@4: {mrr:.3f}")
    state_documents = retriever.retrieve(state_query(STATE, OBJECTIVE), k=4)
    state_hit = any(is_relevant(doc, STATE_RELEVANT) for doc in state_documents)
    print(f"Structured-state parcel prerequisite in top 4: {state_hit}")
    for doc in state_documents:
        print(f"  {doc['score']:.3f} part {doc['part']} {doc['section']} [{doc['id']}]")
    if hit_rate < 0.875 or not state_hit:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
