"""A small deterministic goal-conditioned value for manually supplied goals."""

from collections import deque


def normalize_map_name(name):
    return name.strip().upper().replace(" ", "_")


class LocationValue:
    """Score distance in a supplied directed map graph, or on one map.

    Graph edges are an optimistic, coarse topology: they do not establish that
    routes are traversable under current event flags or within-map collisions.
    """
    def __init__(self, lookups, edges):
        self.maps = lookups["maps"]
        self.edges = {}
        for edge in edges:
            self.edges.setdefault(edge["from"], set()).add(edge["to"])

    def evaluate(self, state, goal):
        if goal.get("goal_type") != "REACH_LOCATION":
            raise NotImplementedError("Only REACH_LOCATION is currently supported")
        target = normalize_map_name(goal["target"])
        current = normalize_map_name(state["map_name"])
        if target not in self.maps or current not in self.maps:
            raise ValueError("Unknown current or target map")
        coords = (goal.get("target_x"), goal.get("target_y"))
        if (coords[0] is None) != (coords[1] is None):
            raise ValueError("target_x and target_y must be supplied together")
        if current == target:
            if coords[0] is None:
                return 1.0
            if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in coords):
                raise ValueError("Target coordinates must be nonnegative integers")
            distance = abs(state["x"] - coords[0]) + abs(state["y"] - coords[1])
            return 1 / (1 + distance)
        if coords[0] is not None:
            raise ValueError("Cross-map tile distances are not supported")
        queue = deque([(current, 0)])
        seen = {current}
        while queue:
            node, distance = queue.popleft()
            for neighbor in self.edges.get(node, ()):
                if neighbor == target:
                    return 1 / (2 + distance)
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, distance + 1))
        raise ValueError("No route in the supplied partial map graph")
