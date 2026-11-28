"""Adaptive initial/best/worst emulator starts for independent PPO experiments.

Each completed episode contributes its *original* return and final emulator
snapshot. A coordinator publishes the highest and lowest eligible episodes from
the most recently completed rollout. Workers see the new pair on their next
reset, without changing their currently running episodes.
"""
import hashlib
import io
import json
import math
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

from pokemon_game.state_encoder import MAP_ID, X_COORD, Y_COORD, PARTY_COUNT, BADGES, EVENT_FLAGS
from pokemon_game.emulator_simulator import BATTLE_FLAG


def role_for_worker(rank, workers):
    """Assign fixed worker roles, giving 50/25/25 when divisible by four."""
    if workers < 4 or workers % 4 or not 0 <= rank < workers:
        raise ValueError('Worker count must be a multiple of four and rank must be valid')
    if rank < workers // 2:
        return 'initial'
    return 'best' if rank < workers * 3 // 4 else 'worst'


def battle_mode(memory):
    """Decode Red's overworld, battle and lost-battle RAM values."""
    flag = memory[BATTLE_FLAG]
    modes = {0: 'overworld', 1: 'wild_battle', 2: 'trainer_battle', 255: 'lost_battle'}
    if flag not in modes:
        raise ValueError(f'Unknown battle flag: {flag}')
    return modes[flag]


def state_info(memory):
    count = memory[PARTY_COUNT]
    if not 0 <= count <= 6:
        raise ValueError(f'Invalid party size: {count}')
    return {'map_id': memory[MAP_ID], 'x': memory[X_COORD], 'y': memory[Y_COORD],
            'party_size': count, 'mode': battle_mode(memory)}


def atomic_write(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(content)
        stream.flush()
    os.replace(temporary, path)


def progress_info(memory):
    """Snapshot raw event flags and party levels for inherited/progress accounting."""
    return {'badges': memory[BADGES].bit_count(),
            'event_flags': [memory[address] for address in range(EVENT_FLAGS, 0xD87E)],
            'party_levels': [memory[0xD18C + i * 44] for i in range(memory[PARTY_COUNT])]}


class SnapshotArchive:
    """Publish snapshots together through one atomic manifest replacement."""
    def __init__(self, directory, excluded_goal_map=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = self.directory / 'current.json'
        self.excluded_goal_map = excluded_goal_map

    def publish(self, candidates, rollout):
        eligible = [candidate for candidate in candidates if candidate.get('eligible')
                    and isinstance(candidate.get('snapshot'), bytes)
                    and candidate['snapshot'] and math.isfinite(candidate['original_return'])
                    and (self.excluded_goal_map is None or
                         candidate.get('state', {}).get('map_id') != self.excluded_goal_map)]
        if not eligible:
            return None
        # Stable ties select the first completion in worker/step order.
        selected = {'best': max(eligible, key=lambda x: x['original_return']),
                    'worst': min(eligible, key=lambda x: x['original_return'])}
        manifest = {'rollout': rollout, 'eligible_episodes': len(eligible), 'states': {}}
        for role, candidate in selected.items():
            data = candidate['snapshot']
            filename = f'{role}-rollout-{rollout:06d}.state'
            atomic_write(self.directory / filename, data)
            manifest['states'][role] = {**{key: value for key, value in candidate.items() if key != 'snapshot'},
                                        'filename': filename, 'sha256': hashlib.sha256(data).hexdigest()}
        atomic_write(self.manifest, (json.dumps(manifest, indent=2) + '\n').encode())
        return manifest


class AdaptiveStarts(gym.Wrapper):
    """Reset from a role's current snapshot; retain the initial-state fallback.

    This wrapper should surround the base environment before any potential
    shaping wrapper, so ranking uses original rewards. Final state validation is
    deliberately minimal: valid party count and known battle flag. It does not
    prove the snapshot is recoverable or distinguish menus from overworld play.
    """
    def __init__(self, env, initial_state, archive_dir, role):
        super().__init__(env)
        if role not in ('initial', 'best', 'worst'):
            raise ValueError('Invalid curriculum role')
        self.initial_state = Path(initial_state).resolve()
        self.archive_dir = Path(archive_dir).resolve()
        self.role = role
        self.original_return = None
        self.steps = 0
        self.source = 'initial'

    def _starting_state(self):
        if self.role == 'initial':
            return self.initial_state, 'initial'
        manifest_path = self.archive_dir / 'current.json'
        if not manifest_path.exists():
            return self.initial_state, 'initial_fallback'
        try:
            manifest = json.loads(manifest_path.read_text())
            entry = manifest['states'][self.role]
            filename = entry['filename']
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError('Invalid archive filename')
            path = self.archive_dir / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Snapshot hash mismatch')
            return path, self.role
        except (OSError, ValueError, TypeError, KeyError):
            return self.initial_state, 'initial_fallback'

    def reset(self, **kwargs):
        path, source = self._starting_state()
        base = self.unwrapped
        old_path = base.init_state
        try:
            base.init_state = str(path)
            try:
                observation, info = self.env.reset(**kwargs)
                initial = state_info(base.pyboy.memory)
            except (OSError, ValueError, RuntimeError):
                if path == self.initial_state:
                    raise
                base.init_state = str(self.initial_state)
                observation, info = self.env.reset(**kwargs)
                initial, source = state_info(base.pyboy.memory), 'initial_fallback'
        finally:
            base.init_state = old_path
        # The base reset initializes these to constants even for arbitrary saves.
        base.last_health = base.read_hp_fraction()
        base.party_size = base.pyboy.memory[PARTY_COUNT]
        # Initialize all screen-history slots from the actually loaded frame.
        base.recent_screens[:] = np.repeat(base.recent_screens[:, :, :1],
                                           base.recent_screens.shape[2], axis=2)
        observation['screens'] = base.recent_screens.copy()
        self.original_return, self.steps, self.source = 0.0, 0, source
        self.initial_state_info = initial
        self.start_progress = progress_info(base.pyboy.memory)
        info = dict(info)
        info['curriculum_start'] = {'role': self.role, 'source': source, **initial}
        return observation, info

    def step(self, action):
        if self.original_return is None:
            raise RuntimeError('Reset required before stepping')
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.original_return += float(reward)
        self.steps += 1
        info = dict(info)
        info['curriculum_transition'] = {'role': self.role, 'source': self.source}
        if terminated or truncated:
            candidate = {'original_return': self.original_return, 'episode_steps': self.steps,
                         'start_role': self.role, 'start_source': self.source,
                         'start_state': self.initial_state_info,
                         'terminated': bool(terminated), 'truncated': bool(truncated)}
            try:
                candidate['state'] = state_info(self.unwrapped.pyboy.memory)
                end = progress_info(self.unwrapped.pyboy.memory)
                start = self.start_progress
                candidate['progress'] = {
                    'start_badges': start['badges'], 'end_badges': end['badges'],
                    'badge_delta': end['badges'] - start['badges'],
                    'start_party_levels': start['party_levels'], 'end_party_levels': end['party_levels'],
                    'raw_event_flags_gained': sum((b & ~a).bit_count() for a, b in zip(start['event_flags'], end['event_flags'])),
                    'raw_event_flags_lost': sum((a & ~b).bit_count() for a, b in zip(start['event_flags'], end['event_flags']))}
                buffer = io.BytesIO()
                self.unwrapped.pyboy.save_state(buffer)
                candidate['snapshot'] = buffer.getvalue()
                candidate['eligible'] = bool(candidate['snapshot']) and math.isfinite(self.original_return)
            except (ValueError, OSError, RuntimeError) as exc:
                candidate['eligible'], candidate['error'] = False, str(exc)
            info['curriculum_episode'] = candidate
            self.original_return = None
        return observation, reward, terminated, truncated, info

    def close(self):
        self.unwrapped.pyboy.stop(save=False)
