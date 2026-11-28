"""Read validated demonstration sessions without Python pickle."""
import json
import hashlib
from pathlib import Path

import numpy as np

SOURCES = frozenset(('human', 'greedy_expert', 'dagger_expert', 'successful_ppo'))


class DemonstrationDataset:
    def __init__(self, sessions, sources=('human', 'greedy_expert', 'dagger_expert'), *, allow_incomplete=False,
                 allow_mixed_goals=False, goal_conditioned=False):
        selected = set(sources)
        if not selected or not selected <= SOURCES:
            raise ValueError('Invalid or empty source filter')
        self.paths, self.sessions, self.goal_by_path = [], [], []
        self.goal_conditioned = goal_conditioned
        self.schema = None
        self.goals = set()
        for session in map(lambda p: Path(p).resolve(), sessions):
            manifest = json.loads((session / 'manifest.json').read_text())
            if manifest.get('format_version') != 1:
                raise ValueError('Unsupported demonstration format')
            source, outcome = manifest.get('source'), manifest.get('outcome')
            if source not in SOURCES:
                raise ValueError('Unknown demonstration source')
            if source not in selected:
                continue
            if outcome == 'recording' and not allow_incomplete:
                raise ValueError(f'Session is still recording: {session}')
            if outcome == 'interrupted' and not allow_incomplete:
                raise ValueError(f'Session did not finish cleanly: {session}')
            if source == 'greedy_expert' and outcome != 'goal_reached':
                raise ValueError('Greedy expert demonstrations must have reached their goal')
            if source == 'successful_ppo' and outcome not in ('success', 'goal_reached'):
                raise ValueError('PPO demonstrations must be marked successful')
            self.goals.add(manifest.get('goal_map_id'))
            if goal_conditioned and manifest.get('goal_map_id') is None:
                raise ValueError('Goal-conditioned demonstrations require a goal label')
            files = sorted((session / 'transitions').glob('*.npz'))
            expected = int(manifest['transitions'])
            if len(files) != expected or [p.name for p in files] != [f'{i:08d}.npz' for i in range(expected)]:
                raise ValueError(f'Incomplete or inconsistent transition files: {session}')
            hashes = manifest.get('transition_hashes')
            if hashes is not None and (len(hashes) != expected or any(
                    hashlib.sha256(path.read_bytes()).hexdigest() != checksum for path, checksum in zip(files, hashes))):
                raise ValueError(f'Demonstration transition hash mismatch: {session}')
            self.sessions.append({'path': str(session), 'source': source, 'outcome': outcome, 'count': expected,
                                  'start_sha256': manifest.get('start_sha256'),
                                  'goal_map_id': manifest.get('goal_map_id')})
            for path in files:
                obs, next_obs, action, metadata = self._read(path, manifest.get('goal_map_id'), goal_conditioned)
                schema = {key: (value.shape, value.dtype.str) for key, value in obs.items()}
                next_schema = {key: (value.shape, value.dtype.str) for key, value in next_obs.items()}
                if not obs or schema != next_schema:
                    raise ValueError(f'Observation and successor schema differ: {path}')
                if self.schema is None:
                    self.schema = schema
                if schema != self.schema:
                    raise ValueError(f'Observation schema differs between transitions: {path}')
                if metadata['source'] != source or metadata['action'] != action or not 0 <= action < 7:
                    raise ValueError(f'Invalid transition metadata: {path}')
                if source == 'dagger_expert' and (not 0 <= action < 4 or
                        type(metadata.get('behavior_action')) is not int or
                        not 0 <= metadata['behavior_action'] < 7):
                    raise ValueError(f'Invalid DAgger behavior or expert label: {path}')
                self.paths.append(path)
                self.goal_by_path.append(manifest.get('goal_map_id'))
        if not self.paths:
            raise ValueError('No demonstrations matched the source filter')
        if len(self.goals) > 1 and not allow_mixed_goals and not goal_conditioned:
            raise ValueError('Mixed or unlabeled goals require explicit opt-in; observations are not goal-conditioned')

    @staticmethod
    def _read(path, goal=None, goal_conditioned=False):
        with np.load(path, allow_pickle=False) as data:
            obs = {key[5:]: np.array(data[key], copy=True) for key in data.files if key.startswith('obs__')}
            successor = {key[6:]: np.array(data[key], copy=True) for key in data.files if key.startswith('next__')}
            action = int(data['action'])
            metadata = json.loads(str(data['metadata'].item()))
        if goal_conditioned:
            from pokemon_game.goal_conditioning import GOAL_KEY, goal_vector
            expected = goal_vector(goal)
            for observation in (obs, successor):
                if GOAL_KEY in observation and not np.array_equal(observation[GOAL_KEY], expected):
                    raise ValueError(f'Observation goal does not match session goal: {path}')
                observation[GOAL_KEY] = expected.copy()
        return obs, successor, action, metadata

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self._read(self.paths[index], self.goal_by_path[index], self.goal_conditioned)

    def batch(self, indices):
        rows = [self[i] for i in indices]
        return ({key: np.stack([row[0][key] for row in rows]) for key in rows[0][0]},
                np.array([row[2] for row in rows], dtype=np.int64))
