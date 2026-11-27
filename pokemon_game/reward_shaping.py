"""Optional fixed-goal potential shaping for the Gymnasium PPO environment."""

import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym

from pokemon_game.state_encoder import MAP_ID, X_COORD, Y_COORD
from pokemon_game.tile_graph import distances_to_target, load_graph


class NavigationShaping(gym.Wrapper):
    """Apply beta * (gamma * Phi(next) - Phi(current)) for one fixed graph.

    Unknown coordinates have zero potential. True terminal states have zero
    potential; time-limit truncations retain the observed state's potential.
    The graph is frozen for the lifetime of the wrapper.
    """
    def __init__(self, env, graph_path, rom_path, beta, gamma):
        super().__init__(env)
        if not math.isfinite(beta) or beta < 0 or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("Invalid shaping beta or gamma")
        data = json.loads(Path(graph_path).read_text())
        context = data["context"]
        if context["rom_sha256"] != hashlib.sha256(Path(rom_path).read_bytes()).hexdigest():
            raise ValueError("Shaping graph belongs to a different ROM")
        graph = load_graph(graph_path, context)
        if graph.target is None or graph.target[0] != context["target_map"]:
            raise ValueError("Shaping graph has no compatible target")
        self.distances = distances_to_target(graph)
        self.target_map = context["target_map"]
        self.beta, self.gamma = beta, gamma
        self.previous_potential = None

    def potential(self):
        memory = self.unwrapped.pyboy.memory
        key = (memory[MAP_ID], memory[X_COORD], memory[Y_COORD])
        if key[0] == self.target_map:
            return 1.0
        distance = self.distances.get(key)
        return 0.0 if distance is None else 1 / (1 + distance)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_potential = self.potential()
        return observation, info

    def step(self, action):
        if self.previous_potential is None:
            raise RuntimeError("Environment must be reset before stepping")
        observation, reward, terminated, truncated, info = self.env.step(action)
        next_potential = 0.0 if terminated else self.potential()
        shaping = self.beta * (self.gamma * next_potential - self.previous_potential)
        self.previous_potential = next_potential
        info = dict(info)
        info["navigation_shaping"] = shaping
        return observation, reward + shaping, terminated, truncated, info
