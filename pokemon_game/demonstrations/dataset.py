"""Read validated demonstration sessions without Python pickle."""
import json
import hashlib
from pathlib import Path

import numpy as np

SOURCES = frozenset(('human', 'greedy_expert', 'successful_ppo'))


class DemonstrationDataset:
    def __init__(self, sessions, sources=('human', 'greedy_expert'), *, allow_incomplete=False,
                 allow_mixed_goals=False):
        selected = set(sources)
        if not selected or not selected <= SOURCES:
            raise ValueError('Invalid or empty source filter')
        self.paths, self.sessions = [], []
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
                obs, next_obs, action, metadata = self._read(path)
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
                self.paths.append(path)
        if not self.paths:
            raise ValueError('No demonstrations matched the source filter')
        if len(self.goals) > 1 and not allow_mixed_goals:
            raise ValueError('Mixed or unlabeled goals require explicit opt-in; observations are not goal-conditioned')

    @staticmethod
    def _read(path):
        with np.load(path, allow_pickle=False) as data:
            obs = {key[5:]: np.array(data[key], copy=True) for key in data.files if key.startswith('obs__')}
            successor = {key[6:]: np.array(data[key], copy=True) for key in data.files if key.startswith('next__')}
            action = int(data['action'])
            metadata = json.loads(str(data['metadata'].item()))
        return obs, successor, action, metadata

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self._read(self.paths[index])

    def batch(self, indices):
        rows = [self[i] for i in indices]
        return ({key: np.stack([row[0][key] for row in rows]) for key in rows[0][0]},
                np.array([row[2] for row in rows], dtype=np.int64))
