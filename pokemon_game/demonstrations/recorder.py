"""Record exact observations and actions from a Gym environment."""
import hashlib
import io
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

from pokemon_game.adaptive_curriculum import battle_mode
from pokemon_game.state_encoder import encode_state


def copy_obs(obs):
    if not isinstance(obs, dict) or not obs:
        raise ValueError('Expected a nonempty dictionary observation')
    return {key: np.array(value, copy=True) for key, value in obs.items()}


def atomic_bytes(path, data):
    temp = Path(str(path) + '.tmp')
    temp.write_bytes(data)
    os.replace(temp, path)


def save_snapshot(pyboy, path):
    buffer = io.BytesIO()
    pyboy.save_state(buffer)
    data = buffer.getvalue()
    atomic_bytes(path, data)
    return hashlib.sha256(data).hexdigest()


class DemonstrationRecorder(gym.Wrapper):
    """Persist each transition atomically; preserve episode boundaries and rewards."""
    def __init__(self, env, output, lookups, source, *, rom_sha256=None, start_sha256=None):
        super().__init__(env)
        if source not in ('human', 'greedy_expert', 'dagger_expert', 'successful_ppo'):
            raise ValueError('Unknown demonstration source')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / 'transitions').mkdir()
        (self.output / 'snapshots').mkdir()
        self.lookups, self.source = lookups, source
        self.manifest = {'format_version': 1, 'source': source, 'rom_sha256': rom_sha256,
                         'start_sha256': start_sha256, 'transitions': 0, 'episodes': 0,
                         'transition_hashes': [], 'snapshots': [], 'outcome': 'recording'}
        self.obs, self.state, self.components = None, None, None
        self.episode, self.episode_step = -1, 0
        self.reward_total = 0.0
        self._write_manifest()

    def _write_manifest(self):
        atomic_bytes(self.output / 'manifest.json', (json.dumps(self.manifest, indent=2) + '\n').encode())

    def _state(self):
        memory = self.unwrapped.pyboy.memory
        state = encode_state(memory, self.lookups)
        state['mode'] = battle_mode(memory)
        return state

    def _components(self):
        return {key: float(value) for key, value in self.unwrapped.progress_reward.items()}

    def snapshot(self, label):
        if self.manifest['outcome'] != 'recording':
            raise RuntimeError('Cannot modify a finalized demonstration')
        filename = f"{len(self.manifest['snapshots']):06d}-{label}.state"
        checksum = save_snapshot(self.unwrapped.pyboy, self.output / 'snapshots' / filename)
        self.manifest['snapshots'].append({'file': filename, 'sha256': checksum, 'label': label,
            'transition': self.manifest['transitions'], 'episode': self.episode})
        self._write_manifest()

    def reset(self, **kwargs):
        if self.manifest['outcome'] != 'recording':
            raise RuntimeError('Cannot modify a finalized demonstration')
        if self.obs is not None:
            raise RuntimeError('Cannot reset before the recorded episode ends')
        obs, info = self.env.reset(**kwargs)
        self.episode += 1
        self.episode_step = 0
        self.manifest['episodes'] += 1
        self.obs, self.state, self.components = copy_obs(obs), self._state(), self._components()
        self.snapshot('episode-start')
        return obs, info

    def step(self, action):
        return self._step(action, action, 'demonstrator')

    def step_labeled(self, behavior_action, expert_action, behavior_source='learner'):
        """Record an expert label while executing the actual roll-in action."""
        if self.source != 'dagger_expert':
            raise ValueError('Separate behavior and expert actions require a DAgger session')
        return self._step(behavior_action, expert_action, behavior_source)

    def _step(self, action, label, behavior_source):
        if self.manifest['outcome'] != 'recording':
            raise RuntimeError('Cannot modify a finalized demonstration')
        if self.obs is None:
            raise RuntimeError('Reset required before recording')
        if not self.action_space.contains(action) or not self.action_space.contains(label):
            raise ValueError('Invalid behavior or expert action')
        before, state_before, components_before = self.obs, self.state, self.components
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        after, state_after, components_after = copy_obs(next_obs), self._state(), self._components()
        original = float(info.get('original_reward', reward))
        delta = {key: components_after.get(key, 0.0) - components_before.get(key, 0.0)
                 for key in set(components_before) | set(components_after)}
        metadata = {'episode': self.episode, 'episode_step': self.episode_step,
            'source': self.source, 'action': int(label), 'behavior_action': int(action),
            'behavior_source': behavior_source, 'reward': original,
            'reward_components': delta, 'terminated': bool(terminated), 'truncated': bool(truncated),
            'state': state_before, 'next_state': state_after, 'goal_reached': bool(info.get('goal_reached', False))}
        arrays = {**{'obs__' + key: value for key, value in before.items()},
                  **{'next__' + key: value for key, value in after.items()},
                  'action': np.array(int(label), dtype=np.int64),
                  'metadata': np.array(json.dumps(metadata, allow_nan=False))}
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        number = self.manifest['transitions']
        data = buffer.getvalue()
        atomic_bytes(self.output / 'transitions' / f'{number:08d}.npz', data)
        self.manifest['transitions'] += 1
        self.manifest['transition_hashes'].append(hashlib.sha256(data).hexdigest())
        self.episode_step += 1
        self.reward_total += original
        self.obs, self.state, self.components = after, state_after, components_after
        if (state_before['map_id'] != state_after['map_id'] or state_before['mode'] != state_after['mode'] or
                state_before['completed_events'] != state_after['completed_events']):
            self.snapshot('milestone')
        if terminated or truncated:
            self.snapshot('episode-end')
            self.obs = None
        self._write_manifest()
        return next_obs, reward, terminated, truncated, info

    def finish(self, outcome):
        if self.manifest['outcome'] != 'recording':
            return
        if self.obs is not None:
            self.snapshot('session-end')
        self.manifest['outcome'] = outcome
        self.manifest['total_original_reward'] = self.reward_total
        self._write_manifest()

    def close(self):
        self.finish('interrupted')
        self.env.close()
