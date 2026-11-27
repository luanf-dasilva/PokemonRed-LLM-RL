"""Resolve supported planner goals to explicit game map identifiers."""

from pokemon_game.value_function import normalize_map_name

ALIASES = {
    "VIRIDIAN_POKÉ_MART": "VIRIDIAN_MART",
    "VIRIDIAN_POKE_MART": "VIRIDIAN_MART",
    "VIRIDIAN_POKÉMON_CENTER": "VIRIDIAN_POKECENTER",
    "VIRIDIAN_POKEMON_CENTER": "VIRIDIAN_POKECENTER",
    "PEWTER_POKÉMON_CENTER": "PEWTER_POKECENTER",
    "PEWTER_POKEMON_CENTER": "PEWTER_POKECENTER",
}


def resolve_goal(goal, lookups):
    if goal.get("goal_type") != "REACH_LOCATION":
        raise ValueError(f"Unsupported goal type: {goal.get('goal_type')}")
    target = goal.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Location target must be a nonempty string")
    name = normalize_map_name(target)
    name = ALIASES.get(name, name)
    if name not in lookups["maps"]:
        raise ValueError(f"Unknown location: {target}")
    return lookups["maps"][name]["id"]


def plan_goal(state, objective, index_path):
    """Retrieve the evaluated multi-query context and call the local planner."""
    from pokemon_rag.planner import PokemonPlanner
    from pokemon_rag.retrieval_experiments import retrieve_context
    from pokemon_rag.retriever import PokemonRetriever

    retriever = PokemonRetriever(index_path)
    documents = retrieve_context(retriever, state, objective, "multi")
    if not documents:
        raise ValueError("No eligible walkthrough context retrieved")
    result = PokemonPlanner(retriever=retriever).generate(state, objective, documents)
    return {"state": state, "objective": objective, "documents": documents, **result}
