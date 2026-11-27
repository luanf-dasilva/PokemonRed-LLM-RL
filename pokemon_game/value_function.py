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


class HierarchicalNavigationValue:
    """Combine observed map transitions and directed tile distances.

    Only sampled transitions that can reach the discovered target are used;
    this keeps disconnected portions of one map distinct. It is a heuristic,
    not a learned return estimate or a full static collision model.
    """
    def __init__(self, discovery, local_weight=0.5):
        from pokemon_game.tile_graph import distances_to_target

        if not 0 < local_weight < 1:
            raise ValueError("local_weight must be between zero and one")
        self.local_weight = local_weight
        self.target_map = discovery.target[0] if discovery.target else None
        reachable = distances_to_target(discovery)
        map_reverse, reverse_tiles, cross_edges = {}, {}, []
        for source, actions in discovery.edges.items():
            if source not in reachable:
                continue
            for target in actions.values():
                if target not in reachable:
                    continue
                if source[0] == target[0]:
                    reverse_tiles.setdefault(target, set()).add(source)
                else:
                    map_reverse.setdefault(target[0], set()).add(source[0])
                    cross_edges.append((source, target))
        self.map_distances = {self.target_map: 0}
        queue = deque([self.target_map])
        while queue:
            current = queue.popleft()
            for predecessor in map_reverse.get(current, ()):
                if predecessor not in self.map_distances:
                    self.map_distances[predecessor] = self.map_distances[current] + 1
                    queue.append(predecessor)
        exits = {source for source, target in cross_edges
                 if self.map_distances.get(source[0]) == self.map_distances.get(target[0], -2) + 1}
        self.tile_distances = {position: 0 for position in exits}
        queue = deque(exits)
        while queue:
            current = queue.popleft()
            for predecessor in reverse_tiles.get(current, ()):
                if predecessor not in self.tile_distances:
                    self.tile_distances[predecessor] = self.tile_distances[current] + 1
                    queue.append(predecessor)

    def evaluate(self, position):
        from pokemon_game.tile_graph import position_key

        key = position_key(position)
        if position.map_id == self.target_map:
            return 1.0
        map_distance = self.map_distances.get(position.map_id)
        local_distance = self.tile_distances.get(key)
        if map_distance is None or local_distance is None:
            return None
        normalized = local_distance / (1 + local_distance)
        return 1 / (1 + map_distance + self.local_weight * normalized)
