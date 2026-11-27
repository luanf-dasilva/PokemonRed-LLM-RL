"""Greedy one-step selection with safe emulator restoration."""

from pokemon_game.emulator_simulator import DIRECTIONS
from pokemon_game.tile_graph import position_key


def select_action(simulator, evaluate):
    """Evaluate each direction from the same snapshot; restore before return."""
    root = simulator.snapshot()
    best_action, best_score = None, float("-inf")
    try:
        for action in range(len(DIRECTIONS)):
            simulator.restore(root)
            successor = simulator.step(action, simulated=True)
            if successor.in_battle:
                continue
            score = evaluate(successor)
            if score is not None and score > best_score:
                best_score, best_action = score, action
    finally:
        simulator.restore(root)
    if best_action is None:
        raise RuntimeError("No eligible action from current state")
    return best_action


def tile_value(distances, successor):
    distance = distances.get(position_key(successor))
    return None if distance is None else 1 / (1 + distance)


def select_model_action(graph, position, distances):
    """Choose an observed edge that strictly decreases directed distance."""
    current = position_key(position)
    distance = distances.get(current)
    if distance is None:
        raise RuntimeError("Current position has no known path to the target")
    candidates = [(distances[target], action) for action, target in graph.edges.get(current, {}).items()
                  if target in distances and distances[target] < distance]
    if not candidates:
        raise RuntimeError("No known improving transition")
    return min(candidates)[1]


def correct_transition(graph, source, action, predicted, observed, allowed_maps):
    """Repair a prediction from live observations; battle edges remain unknown."""
    target = position_key(observed)
    if not observed.in_battle and target == predicted:
        return False
    graph.edges.get(source, {}).pop(action, None)
    if not observed.in_battle and target != source and target[0] in allowed_maps:
        graph.edges.setdefault(source, {})[action] = target
    return True
