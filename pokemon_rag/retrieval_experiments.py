"""Retrieval variants for controlled expanded-corpus experiments."""

from pokemon_rag.planner import retrieval_query


def generation_one_verified(document):
    """Use only source-scoped Red/Blue walkthroughs or curated sourced facts.

    Mixed-scope articles remain indexed but are not assumed to apply to Gen I.
    This is a conservative provenance filter, not a proof that every source
    statement is correct.
    """
    if (document.get("generation") != 1 or not document.get("source")
            or document.get("game") != "Pokémon Red and Blue"):
        return False
    scope = document.get("scope")
    if document.get("site") == "bulbapedia" and scope == "generation I walkthrough":
        return True
    return (document.get("site") == "curated"
            and scope == "manually authored source-grounded knowledge"
            and bool(document.get("source_section")))


def retrieve_context(retriever, state, objective, mode):
    predicate = generation_one_verified
    if mode in {"top4", "top8"}:
        k = 4 if mode == "top4" else 8
        return retriever.retrieve(retrieval_query(state, objective), k=k, predicate=predicate)
    if mode != "multi":
        raise ValueError(f"Unknown retrieval mode: {mode}")
    queries = [
        f"Pokemon Red {state['location']} walkthrough and progression",
        f"Pokemon Red {objective} requirements",
        f"Pokemon Red {state['location']} objectives and available activities",
    ]
    groups = [retriever.retrieve(query, k=4, predicate=predicate) for query in queries]
    selected, seen = [], set()
    # Round-robin ensures local, objective and activity queries get early slots.
    for rank in range(4):
        for group in groups:
            if rank < len(group) and group[rank]["id"] not in seen:
                doc = group[rank]
                selected.append(doc)
                seen.add(doc["id"])
                if len(selected) == 8:
                    return selected
    return selected
