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
