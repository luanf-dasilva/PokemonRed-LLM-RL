"""Discover a directed tile graph by probing isolated emulator snapshots."""

from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

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


def save_graph(path, discovery, context):
    """Atomically store a graph together with its exact compatibility context."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "context": context, "start": discovery.start,
            "target": discovery.target, "explored_nodes": discovery.explored_nodes,
            "truncated": discovery.truncated,
            "edges": [{"from": source, "actions": {str(a): target for a, target in actions.items()}}
                      for source, actions in discovery.edges.items()]}
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            name = stream.name
            json.dump(data, stream, sort_keys=True)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def load_graph(path, context):
    """Reject stale or malformed caches instead of using them optimistically."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or data.get("context") != context:
        raise ValueError("Graph cache is incompatible with this run")
    def position(value):
        if (not isinstance(value, list) or len(value) != 3 or
                any(type(item) is not int or not 0 <= item <= 255 for item in value)):
            raise ValueError("Invalid cached position")
        return tuple(value)
    edges = {}
    for row in data["edges"]:
        source = position(row["from"])
        if source in edges:
            raise ValueError("Duplicate cached position")
        actions = {}
        for key, target in row["actions"].items():
            if key not in {"0", "1", "2", "3"}:
                raise ValueError("Invalid cached action")
            actions[int(key)] = position(target)
        edges[source] = actions
    target = position(data["target"]) if data["target"] is not None else None
    if type(data["explored_nodes"]) is not int or data["explored_nodes"] < 1 or type(data["truncated"]) is not bool:
        raise ValueError("Invalid graph metadata")
    return Discovery(edges, position(data["start"]), target, data["explored_nodes"], data["truncated"])
