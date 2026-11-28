"""Goal-conditioned map navigation observations and episode termination."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pokemon_game.state_encoder import MAP_ID
from pokemon_game.emulator_simulator import BATTLE_FLAG

GOAL_KEY = 'goal_map'


def goal_vector(goal):
    if isinstance(goal, bool) or not isinstance(goal, (int, np.integer)) or not 0 <= int(goal) <= 255:
        raise ValueError('Goal map must be an integer between 0 and 255')
    result = np.zeros(256, dtype=np.float32)
    result[int(goal)] = 1.0
    return result


class GoalConditionedMap(gym.Wrapper):
    """Choose one map goal at reset and expose it throughout the episode.

    Goals already satisfied at reset are excluded; if none remain, reset fails
    explicitly. Goals are held fixed within an episode, and only reaching the
    map outside battle causes goal termination. This wrapper does not claim that
    every supplied goal is reachable from every possible starting state.
    """
    def __init__(self, env, goal_maps):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError('Goal conditioning requires dictionary observations')
        supplied = tuple(goal_maps)
        for goal in supplied:
            goal_vector(goal)
        goals = tuple(int(x) for x in supplied)
        if not goals or len(set(goals)) != len(goals):
            raise ValueError('Provide a nonempty list of distinct goal maps')
        if GOAL_KEY in env.observation_space.spaces:
            raise ValueError('Observation already contains the goal key')
        self.goals = goals
        self.observation_space = spaces.Dict({**env.observation_space.spaces,
                                              GOAL_KEY: spaces.Box(0.0, 1.0, shape=(256,), dtype=np.float32)})
        self.rng = np.random.default_rng()
        self.goal = None

    def _observation(self, observation):
        return {**observation, GOAL_KEY: goal_vector(self.goal)}

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        observation, info = self.env.reset(seed=seed, options=options)
        current_map = self.unwrapped.pyboy.memory[MAP_ID]
        eligible = [goal for goal in self.goals if goal != current_map]
        requested = None if options is None else options.get('goal_map_id')
        if requested is not None:
            goal_vector(requested)
            if requested not in eligible:
                raise ValueError('Requested goal is unavailable or already satisfied')
            self.goal = int(requested)
        elif not eligible:
            raise ValueError('No configured goal remains after excluding the current map')
        else:
            self.goal = eligible[int(self.rng.integers(len(eligible)))]
        info = dict(info)
        info['goal_map_id'] = self.goal
        return self._observation(observation), info

    def step(self, action):
        if self.goal is None:
            raise RuntimeError('Reset required before stepping')
        obs, reward, terminated, truncated, info = self.env.step(action)
        memory = self.unwrapped.pyboy.memory
        reached = memory[MAP_ID] == self.goal and not memory[BATTLE_FLAG]
        info = dict(info)
        info['goal_map_id'], info['goal_reached'] = self.goal, bool(reached)
        return self._observation(obs), reward, terminated or reached, truncated and not reached, info
