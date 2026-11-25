"""Generate a structured high-level Pokémon Red goal using local Ollama."""

import json
import time

import requests
from pydantic import ValidationError

from pokemon_rag.schemas import Goal

MODEL = "qwen3.5:4b-q4_K_M"
OLLAMA_URL = "http://localhost:11434/api/chat"
SYSTEM_PROMPT = """You are a hierarchical planner for the original Game Boy Pokémon Red.
Your task is to select ONE achievable intermediate objective based on the
current game state. You are not responsible for solving the entire long-term
objective in one response.

Decision procedure:
1. Examine the current game state.
2. Identify any incomplete mandatory prerequisites.
3. Consider the party's health and readiness.
4. Select the next achievable objective supported by the available information.
5. Provide a measurable completion condition.

An intermediate objective does not need to directly complete the long-term
objective. It is acceptable to select a goal that temporarily moves away from
the final destination when completing a prerequisite or recovering health.
Do not require evidence describing the entire path to the long-term objective.
Select INSUFFICIENT_CONTEXT only when you cannot identify any reasonable
intermediate objective.

When walkthrough context is supplied, cite relevant chunk IDs supporting the
selected goal. If no walkthrough context is supplied, use an empty evidence list.
Do not invent game mechanics or map connections. Do not provide individual
controller actions. The completion_condition is a plain-language description,
not code. Return the requested structured output."""


def retrieval_query(state, objective):
    return (
        f"Current location: {state['location']}. Objective: {objective}. "
        f"Current game state: {json.dumps(state)}. "
        "What prerequisite or next step is required?"
    )


def make_payload(state, objective, documents, model=MODEL, placement="state_first"):
    context = "\n\n".join(
        f"[{doc['id']}]\nSection: {doc['section']}\n{doc['text']}"
        for doc in documents
    )
    state_text = f"CURRENT STATE:\n{json.dumps(state, indent=2, ensure_ascii=False)}"
    objective_text = f"LONG-TERM OBJECTIVE:\n{objective}"
    context_text = f"WALKTHROUGH CONTEXT:\n{context}"
    if placement == "state_first":
        prompt = "\n\n".join((state_text, objective_text, context_text))
    elif placement == "state_last":
        prompt = "\n\n".join((context_text, objective_text, state_text))
    else:
        raise ValueError(f"Unknown prompt placement: {placement}")
    return {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "format": Goal.model_json_schema(),
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }


class PokemonPlanner:
    def __init__(self, retriever=None, model=MODEL, url=OLLAMA_URL, timeout=180):
        self.retriever = retriever
        self.model = model
        self.url = url
        self.timeout = timeout

    def get_context(self, state, objective, k=4):
        if self.retriever is None:
            from pokemon_rag.retriever import PokemonRetriever
            self.retriever = PokemonRetriever()
        documents = self.retriever.retrieve(retrieval_query(state, objective), k=k)
        if not documents:
            raise RuntimeError("No walkthrough context retrieved")
        return documents

    def generate(self, state, objective, documents, placement="state_first"):
        """Return raw and validated output plus timing and citation diagnostics."""
        payload = make_payload(state, objective, documents, self.model, placement=placement)
        started = time.perf_counter()
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        latency = time.perf_counter() - started
        response.raise_for_status()
        result = response.json()
        content = result.get("message", {}).get("content", "")
        try:
            goal = Goal.model_validate_json(content)
            error = None
        except (ValidationError, ValueError) as exc:
            goal = None
            error = str(exc)
        ids = {doc["id"] for doc in documents}
        citations_valid = None if goal is None else all(item in ids for item in goal.evidence)
        evidence_present = None if goal is None else bool(goal.evidence)
        return {
            "request": payload,
            "response": result,
            "raw_content": content,
            "goal": goal.model_dump() if goal else None,
            "schema_valid": goal is not None,
            "validation_error": error,
            "citation_ids_valid": citations_valid,
            "evidence_present": evidence_present,
            "latency_seconds": latency,
        }

    def plan(self, state, objective, use_rag=True):
        documents = self.get_context(state, objective) if use_rag else []
        result = self.generate(state, objective, documents)
        if not result["schema_valid"]:
            raise ValueError(f"Invalid planner response: {result['validation_error']}")
        if not result["citation_ids_valid"]:
            raise ValueError("Planner cited a chunk that was not supplied")
        if use_rag and not result["evidence_present"] and result["goal"]["goal_type"] != "INSUFFICIENT_CONTEXT":
            raise ValueError("Planner did not cite evidence for its goal")
        return Goal.model_validate(result["goal"])


if __name__ == "__main__":
    state = {"location": "Viridian City", "party_size": 1, "party_levels": [7],
             "hp_fraction": 0.8, "has_oaks_parcel": False,
             "delivered_oaks_parcel": False, "badges": 0}
    print(PokemonPlanner().plan(state, "Obtain the Boulder Badge").model_dump_json(indent=2))
