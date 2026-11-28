import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np

from pokemon_game.adaptive_curriculum import AdaptiveStarts, SnapshotArchive, battle_mode, role_for_worker
from pokemon_game.reward_shaping import NavigationShaping
from pokemon_game.tile_graph import Discovery, save_graph


class FakeBoy:
    def __init__(self):
        self.memory = bytearray(65536)

    def save_state(self, stream):
        stream.write(json.dumps({'map': self.memory[0xD35E], 'x': self.memory[0xD362],
                                 'party': self.memory[0xD163], 'battle': self.memory[0xD057],
                                 'hp': self.memory[0xD16C]}).encode())

    def stop(self, save=False):
        pass


class FakeEnv(gym.Env):
    def __init__(self, path):
        self.init_state = str(path)
        self.pyboy = FakeBoy()

    def reset(self, **kwargs):
        data = json.loads(Path(self.init_state).read_bytes())
        mem = self.pyboy.memory
        for key, address in [('map', 0xD35E), ('x', 0xD362), ('party', 0xD163),
                             ('battle', 0xD057), ('hp', 0xD16C)]:
            mem[address] = data[key]
        self.last_health, self.party_size = 1.0, 0
        self.base_event_flags = data.get('events', 0)
        self.step_count = 0
        self.recent_screens = np.zeros((2, 2, 3), dtype=np.uint8)
        self.recent_screens[:, :, 0] = data['x'] + 5
        return {'screens': self.recent_screens.copy()}, {}

    def read_hp_fraction(self):
        return self.pyboy.memory[0xD16C] / 10

    def step(self, action):
        reward, x, battle, party = action
        self.pyboy.memory[0xD362] = x
        self.pyboy.memory[0xD057] = battle
        self.pyboy.memory[0xD163] = party
        self.step_count += 1
        return {'screens': self.recent_screens.copy()}, reward, False, True, {}


def state(path, x=0, hp=4, party=1, battle=0):
    path.write_text(json.dumps({'map': 0, 'x': x, 'party': party, 'battle': battle, 'hp': hp, 'events': 7}))


class CurriculumTests(unittest.TestCase):
    def test_32_worker_allocation_and_battle_modes(self):
        roles = [role_for_worker(i, 32) for i in range(32)]
        self.assertEqual((roles.count('initial'), roles.count('best'), roles.count('worst')), (16, 8, 8))
        with self.assertRaises(ValueError):
            role_for_worker(0, 3)
        memory = bytearray(65536)
        for flag, expected in enumerate(('overworld', 'wild_battle', 'trainer_battle')):
            memory[0xD057] = flag
            self.assertEqual(battle_mode(memory), expected)
        memory[0xD057] = 255
        self.assertEqual(battle_mode(memory), 'lost_battle')
        memory[0xD057] = 3
        with self.assertRaises(ValueError):
            battle_mode(memory)

    def test_best_worst_publication_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / 'initial.state'
            state(initial)
            archive = SnapshotArchive(root / 'archive')
            worker = AdaptiveStarts(FakeEnv(initial), initial, archive.directory, 'best')
            obs, info = worker.reset()
            self.assertEqual(info['curriculum_start']['source'], 'initial_fallback')
            self.assertEqual(worker.unwrapped.last_health, 0.4)
            self.assertEqual(worker.unwrapped.party_size, 1)
            self.assertTrue(np.all(obs['screens'] == 5))
            _, _, _, _, high = worker.step((5.0, 3, 0, 1))
            worker.reset()
            _, _, _, _, low = worker.step((-2.0, 8, 1, 1))
            best = high['curriculum_episode']
            worst = low['curriculum_episode']
            published = archive.publish([worst, best], rollout=1)
            self.assertEqual(published['states']['best']['original_return'], 5.0)
            self.assertEqual(published['states']['worst']['original_return'], -2.0)
            obs, info = worker.reset()
            self.assertEqual(info['curriculum_start']['source'], 'best')
            self.assertEqual(info['curriculum_start']['x'], 3)
            self.assertTrue(np.all(obs['screens'] == 8))
            other = AdaptiveStarts(FakeEnv(initial), initial, archive.directory, 'worst')
            _, info = other.reset()
            self.assertEqual(info['curriculum_start']['x'], 8)
            self.assertEqual(info['curriculum_start']['mode'], 'wild_battle')
            (archive.directory / published['states']['best']['filename']).write_bytes(b'corrupt')
            _, info = worker.reset()
            self.assertEqual(info['curriculum_start']['source'], 'initial_fallback')
            self.assertIsNone(archive.publish([], rollout=2))
            self.assertEqual(json.loads(archive.manifest.read_text())['rollout'], 1)

    def test_invalid_candidate_is_not_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / 'initial.state'
            state(initial)
            archive = SnapshotArchive(root / 'archive')
            worker = AdaptiveStarts(FakeEnv(initial), initial, archive.directory, 'initial')
            worker.reset()
            _, _, _, _, info = worker.step((100, 3, 9, 1))
            self.assertFalse(info['curriculum_episode']['eligible'])
            self.assertIsNone(archive.publish([info['curriculum_episode']], 1))
            self.assertFalse(archive.manifest.exists())

    def test_goal_snapshots_are_excluded_and_progress_is_episode_local(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / 'initial.state'
            state(initial)
            archive = SnapshotArchive(root / 'archive', excluded_goal_map=1)
            worker = AdaptiveStarts(FakeEnv(initial), initial, archive.directory, 'initial')
            worker.reset()
            worker.unwrapped.pyboy.memory[0xD356] = 1
            worker.unwrapped.pyboy.memory[0xD747] = 2
            _, _, _, _, info = worker.step((3.0, 1, 0, 1))
            candidate = info['curriculum_episode']
            self.assertEqual(candidate['progress']['badge_delta'], 1)
            self.assertEqual(candidate['progress']['raw_event_flags_gained'], 1)
            self.assertEqual(info['curriculum_transition'], {'role': 'initial', 'source': 'initial'})
            excluded = {**candidate, 'state': {**candidate['state'], 'map_id': 1}, 'original_return': 99}
            published = archive.publish([excluded, candidate], 1)
            self.assertEqual(published['eligible_episodes'], 1)
            self.assertEqual(published['states']['best']['original_return'], 3)

    def test_shaping_reinitializes_after_adaptive_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / 'initial.state'
            state(initial)
            archive = SnapshotArchive(root / 'archive')
            base = AdaptiveStarts(FakeEnv(initial), initial, archive.directory, 'best')
            rom, graph = root / 'rom.gb', root / 'graph.json'
            rom.write_bytes(b'rom')
            edges = {(0, 0, 0): {2: (0, 3, 0)}, (0, 3, 0): {2: (1, 0, 0)}}
            context = {'rom_sha256': hashlib.sha256(b'rom').hexdigest(), 'target_map': 1}
            save_graph(graph, Discovery(edges, (0, 0, 0), (1, 0, 0), 3, False), context)
            env = NavigationShaping(base, graph, rom, 1, 0.9)
            env.reset()
            _, _, _, _, info = env.step((1.0, 3, 0, 1))
            archive.publish([info['curriculum_episode']], 1)
            _, info = env.reset()
            self.assertEqual(info['curriculum_start']['source'], 'best')
            self.assertAlmostEqual(env.previous_potential, 0.5)

    def test_fixed_goal_terminates_and_rejects_already_completed_start(self):
        from pokemon_game.curriculum_experiment import FixedGoal

        with tempfile.TemporaryDirectory() as directory:
            initial = Path(directory) / 'initial.state'
            state(initial)
            class GoalEnv(FakeEnv):
                def step(self, action):
                    obs, reward, terminated, truncated, info = super().step(action)
                    self.pyboy.memory[0xD35E] = 1
                    return obs, reward, terminated, truncated, info
            env = FixedGoal(GoalEnv(initial), 1)
            env.reset()
            _, _, terminated, truncated, info = env.step((1.0, 2, 0, 1))
            self.assertTrue(info['goal_reached'])
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            data = json.loads(initial.read_text())
            data['map'] = 1
            initial.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'already on the goal'):
                env.reset()


if __name__ == '__main__':
    unittest.main()
