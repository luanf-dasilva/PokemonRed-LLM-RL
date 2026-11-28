"""Create the same seven-action v2 Gym environment used for PPO."""
import os
import sys
from pathlib import Path

import gymnasium as gym

ROOT = Path(__file__).resolve().parents[2]


class ManagedRedEnv(gym.Wrapper):
    def close(self):
        self.unwrapped.pyboy.stop(save=False)


class StopAtMap(gym.Wrapper):
    """Optional fixed goal with the same terminal semantics as navigation PPO."""
    def __init__(self, env, target_map):
        super().__init__(env)
        self.target_map = target_map

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        memory = self.unwrapped.pyboy.memory
        success = memory[0xD35E] == self.target_map and not memory[0xD057]
        info = dict(info)
        info['goal_reached'] = bool(success)
        return obs, reward, terminated or success, truncated and not success, info


def make_env(rom, start_state, session_dir, max_steps=300, target_map=None):
    """Create a raw v2 env; callers must run from v2 for its events.json."""
    sys.path.insert(0, str(ROOT / 'v2'))
    from red_gym_env_v2 import RedGymEnv
    config = {'headless': True, 'save_final_state': False, 'early_stop': False,
        'action_freq': 24, 'init_state': str(Path(start_state).resolve()), 'max_steps': max_steps,
        'print_rewards': False, 'save_video': False, 'fast_video': True,
        'session_path': Path(session_dir).resolve(), 'gb_path': str(Path(rom).resolve()),
        'debug': False, 'reward_scale': 0.5, 'explore_weight': 0.25,
        'rewards': {'level': True, 'opponent_level': True, 'death_penalty': True}}
    config['session_path'].mkdir(parents=True, exist_ok=True)
    env = ManagedRedEnv(RedGymEnv(config))
    env.unwrapped.pyboy.set_emulation_speed(0)
    return StopAtMap(env, target_map) if target_map is not None else env


def enter_v2():
    os.chdir(ROOT / 'v2')
