"""Fixed-goal potential shaping for Gymnasium environments."""

import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym

from pokemon_game.state_encoder import MAP_ID, X_COORD, Y_COORD
from pokemon_game.tile_graph import distances_to_target, load_graph


class PotentialShaping(gym.Wrapper):
    """Apply beta * (gamma * Phi(next) - Phi(current)).

    The supplied potential returns (value, missing) for a map/tile tuple.
    True terminals have potential zero, while truncations retain next-state
    potential so value bootstrapping remains valid. Each wrapper owns its own
    previous potential and never changes the fixed potential during an episode.
    """
    def __init__(self, env, potential, beta, gamma):
        super().__init__(env)
        if not math.isfinite(beta) or beta < 0 or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("Invalid shaping beta or gamma")
        self.potential_fn = potential
        self.beta, self.gamma = beta, gamma
        self.previous_potential = None

    def position(self):
        memory = self.unwrapped.pyboy.memory
        return (memory[MAP_ID], memory[X_COORD], memory[Y_COORD])

    def potential(self):
        return self.potential_fn(self.position())[0]

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_potential, missing = self.potential_fn(self.position())
        info = dict(info)
        info['navigation_potential_missing'] = missing
        return observation, info

    def step(self, action):
        if self.previous_potential is None:
            raise RuntimeError("Environment must be reset before stepping")
        observation, reward, terminated, truncated, info = self.env.step(action)
        next_potential, missing = (0.0, False) if terminated else self.potential_fn(self.position())
        shaping = self.beta * (self.gamma * next_potential - self.previous_potential)
        self.previous_potential = next_potential
        info = dict(info)
        info['original_reward'] = float(reward)
        info['navigation_shaping'] = float(shaping)
        info['navigation_potential_missing'] = bool(missing)
        if terminated or truncated:
            self.previous_potential = None
        return observation, reward + shaping, terminated, truncated, info


class TilePotential:
    def __init__(self, graph_path, rom_path):
        data = json.loads(Path(graph_path).read_text())
        context = data['context']
        if context['rom_sha256'] != hashlib.sha256(Path(rom_path).read_bytes()).hexdigest():
            raise ValueError('Shaping graph belongs to a different ROM')
        graph = load_graph(graph_path, context)
        if graph.target is None or graph.target[0] != context['target_map']:
            raise ValueError('Shaping graph has no compatible target')
        self.distances = distances_to_target(graph)
        self.target_map = context['target_map']

    def __call__(self, key):
        if key[0] == self.target_map:
            return 1.0, False
        distance = self.distances.get(key)
        return (0.0, True) if distance is None else (1 / (1 + distance), False)


class MapPotential:
    def __init__(self, values):
        self.values = dict(values)

    def __call__(self, key):
        value = self.values.get(key[0])
        return (0.0, True) if value is None else (value, False)


class NavigationShaping(PotentialShaping):
    """Backward-compatible tile-graph shaping wrapper for fixed navigation."""
    def __init__(self, env, graph_path, rom_path, beta, gamma):
        super().__init__(env, TilePotential(graph_path, rom_path), beta, gamma)
