"""Discover a directed tile graph by probing isolated emulator snapshots."""

from collections import deque
from dataclasses import dataclass

from pokemon_game.emulator_simulator import DIRECTIONS


def manhattan_distance(position, target):
    return abs(position[0] - target[0]) + abs(position[1] - target[1])


@dataclass
class Discovery:
    edges: dict
    start: tuple
    target: tuple | None
    explored_nodes: int
    truncated: bool


def position_key(position):
    return (position.map_id, position.x, position.y)


def discover(simulator, target_map_id, allowed_maps, max_nodes=1000):
    """Explore sampled state transitions without committing emulator actions.

    One emulator snapshot is retained per map/tile, so NPC and RNG variations
    are not exhaustively represented. Battle states are excluded. Missing edges
    mean unknown or blocked in the sampled snapshot, not permanently impassable.
    """
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    root = simulator.snapshot()
    initial = simulator.observe()
    if initial.in_battle or initial.map_id not in allowed_maps:
        raise ValueError("Initial state must be on an allowed map and outside battle")
    start = position_key(initial)
    snapshots = {start: root}
    queue = deque([start])
    edges = {}
    goal = start if initial.map_id == target_map_id else None
    truncated = False
    try:
        while queue and goal is None:
            current = queue.popleft()
            edges.setdefault(current, {})
            for action in range(len(DIRECTIONS)):
                simulator.restore(snapshots[current])
                after = simulator.step(action, simulated=True)
                target = position_key(after)
                if after.in_battle or after.map_id not in allowed_maps or target == current:
                    continue
                edges[current][action] = target
                if target not in snapshots:
                    if len(snapshots) >= max_nodes:
                        truncated = True
                        continue
                    snapshots[target] = simulator.snapshot()
                    queue.append(target)
                if after.map_id == target_map_id:
                    goal = target
                    break
    finally:
        simulator.restore(root)
    return Discovery(edges, start, goal, len(snapshots), truncated)


def distances_to_target(discovery):
    """Compute directed BFS distances over observed transitions."""
    if discovery.target is None:
        raise ValueError("Target was not discovered")
    reverse = {}
    for source, actions in discovery.edges.items():
        for target in actions.values():
            reverse.setdefault(target, set()).add(source)
    distances = {discovery.target: 0}
    queue = deque([discovery.target])
    while queue:
        target = queue.popleft()
        for predecessor in reverse.get(target, ()):
            if predecessor not in distances:
                distances[predecessor] = distances[target] + 1
                queue.append(predecessor)
    return distances
