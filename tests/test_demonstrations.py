"""Tests for demonstration integrity and PPO/BC update boundaries."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pokemon_game.demonstrations.dataset import DemonstrationDataset
from pokemon_game.demonstrations.human_player import resolve_start
from pokemon_game.demonstrations.recorder import DemonstrationRecorder


EVENTS = ('EVENT_GOT_STARTER', 'EVENT_GOT_OAKS_PARCEL', 'EVENT_OAK_GOT_PARCEL',
          'EVENT_GOT_POKEDEX', 'EVENT_BEAT_BROCK')
LOOKUPS = {'maps': {'PALLET_TOWN': {'id': 0}, 'ROUTE_1': {'id': 1}},
           'species': {}, 'items': {}, 'events': {name: {'id': i} for i, name in enumerate(EVENTS)}}


class FakeBoy:
    def __init__(self):
        self.memory = bytearray(65536)

    def save_state(self, stream):
        stream.write(bytes(self.memory))


class FakeEnv(gym.Env):
    observation_space = spaces.Dict({'frame': spaces.Box(0, 255, shape=(2,), dtype=np.uint8)})
    action_space = spaces.Discrete(7)

    def __init__(self):
        self.pyboy = FakeBoy()
        self.progress_reward = {'event': 0.0, 'explore': 0.0}
        self.count = 0

    def reset(self, **kwargs):
        self.pyboy.memory[0xD35E] = 0
        self.pyboy.memory[0xD747] = 0
        self.progress_reward = {'event': 0.0, 'explore': 0.0}
        self.count = 0
        return {'frame': np.array([0, 1], dtype=np.uint8)}, {}

    def step(self, action):
        self.count += 1
        self.pyboy.memory[0xD35E] = 1
        self.pyboy.memory[0xD747] = 1
        self.progress_reward['event'] += 1.0
        return {'frame': np.array([self.count, 1], dtype=np.uint8)}, 2.0, False, self.count == 2, {'original_reward': 1.0}


def session(path, source='human', goal=None, start='initial'):
    rec = DemonstrationRecorder(FakeEnv(), path, LOOKUPS, source, start_sha256=start)
    rec.manifest['goal_map_id'] = goal
    rec.reset()
    rec.step(2)
    rec.step(3)
    rec.finish('goal_reached' if source == 'greedy_expert' else 'human_stopped')
    rec.close()
    return path


class RecorderDatasetTests(unittest.TestCase):
    def test_records_exact_transitions_and_original_components(self):
        with tempfile.TemporaryDirectory() as directory:
            path = session(Path(directory) / 'demo')
            dataset = DemonstrationDataset([path])
            self.assertEqual(len(dataset), 2)
            obs, after, action, metadata = dataset[0]
            np.testing.assert_array_equal(obs['frame'], [0, 1])
            np.testing.assert_array_equal(after['frame'], [1, 1])
            self.assertEqual(action, 2)
            self.assertEqual(metadata['reward'], 1)
            self.assertEqual(metadata['reward_components']['event'], 1)
            self.assertEqual(metadata['state']['map_id'], 0)
            self.assertEqual(metadata['next_state']['map_id'], 1)
            self.assertEqual(metadata['next_state']['completed_events'], ['EVENT_GOT_STARTER'])
            self.assertFalse(metadata['truncated'])
            self.assertTrue(dataset[1][3]['truncated'])
            manifest = json.loads((path / 'manifest.json').read_text())
            self.assertEqual([s['label'] for s in manifest['snapshots']], ['episode-start', 'milestone', 'episode-end'])
            self.assertEqual(manifest['total_original_reward'], 2)
            transition = path / 'transitions' / '00000000.npz'
            transition.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                DemonstrationDataset([path])

    def test_goal_filter_and_split_prevent_ambiguous_demos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = session(root / 'one', 'greedy_expert', 1)
            second = session(root / 'two', 'greedy_expert', 2)
            with self.assertRaisesRegex(ValueError, 'Mixed or unlabeled goals'):
                DemonstrationDataset([first, second])
            self.assertEqual(len(DemonstrationDataset([first, second], allow_mixed_goals=True)), 4)
            conditioned = DemonstrationDataset([first, second], goal_conditioned=True)
            self.assertEqual(conditioned[0][0]['goal_map'].argmax(), 1)
            self.assertEqual(conditioned[2][0]['goal_map'].argmax(), 2)
            self.assertEqual(conditioned[2][1]['goal_map'].sum(), 1)
            from pokemon_game.demonstrations.behavioral_cloning import validate_split
            a, b = DemonstrationDataset([first]), DemonstrationDataset([first])
            with self.assertRaisesRegex(ValueError, 'share an exact starting snapshot'):
                validate_split(a, b)

    def test_goal_conditioning_changes_observation_and_terminates_at_selected_goal(self):
        from pokemon_game.goal_conditioning import GoalConditionedMap
        env = GoalConditionedMap(FakeEnv(), [1, 2])
        obs, info = env.reset(seed=7, options={'goal_map_id': 2})
        self.assertEqual(info['goal_map_id'], 2)
        self.assertEqual(obs['goal_map'].argmax(), 2)
        obs, _, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated or truncated)
        self.assertFalse(info['goal_reached'])
        obs, info = env.reset(options={'goal_map_id': 1})
        self.assertEqual(obs['goal_map'].argmax(), 1)
        obs, _, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['goal_reached'])
        with self.assertRaisesRegex(ValueError, 'unavailable or already satisfied'):
            GoalConditionedMap(FakeEnv(), [0]).reset(options={'goal_map_id': 0})

    def test_dagger_records_expert_label_not_behavior_action(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dagger'
            recorder = DemonstrationRecorder(FakeEnv(), path, LOOKUPS, 'dagger_expert', start_sha256='different')
            recorder.manifest['goal_map_id'] = 1
            recorder.reset()
            recorder.step_labeled(behavior_action=3, expert_action=2)
            recorder.finish('expert_abstained')
            recorder.close()
            dataset = DemonstrationDataset([path])
            _, _, label, metadata = dataset[0]
            self.assertEqual(label, 2)
            self.assertEqual(metadata['behavior_action'], 3)
            self.assertEqual(metadata['behavior_source'], 'learner')

    def test_dagger_oracle_abstains_after_invalidating_wrong_edge(self):
        from pokemon_game.demonstrations.dagger import ExpertOracle
        from pokemon_game.emulator_simulator import Position
        from pokemon_game.tile_graph import Discovery

        class Probe:
            def restore(self, snapshot): pass
            def step(self, action, simulated=False): return Position(0, 0, 0, False)

        graph = Discovery({(0, 0, 0): {2: (1, 0, 0)}}, (0, 0, 0), (1, 0, 0), 2, False)
        oracle = ExpertOracle(Probe(), graph, {0, 1}, 1, max_nodes=10, max_discoveries=0)
        self.assertIsNone(oracle.label(b'state', Position(0, 0, 0, False)))
        self.assertEqual(oracle.mismatches, 1)
        self.assertEqual(graph.edges[(0, 0, 0)], {})

    def test_dagger_recovery_verifies_a_known_edge_before_labeling(self):
        from pokemon_game.demonstrations.dagger import ExpertOracle
        from pokemon_game.emulator_simulator import Position
        from pokemon_game.tile_graph import Discovery

        class Probe:
            def __init__(self): self.position, self.calls = (40, 0, 0), 0
            def snapshot(self): return self.position
            def restore(self, snapshot): self.position = snapshot
            def step(self, action, simulated=False):
                self.calls += 1
                if self.position == (40, 0, 0) and action == 0:
                    self.position = (0, 1, 0)
                elif self.position == (0, 1, 0) and action == 2:
                    self.position = (1, 0, 0)
                return Position(*self.position, False)

        graph = Discovery({(0, 1, 0): {2: (1, 0, 0)}}, (0, 1, 0), (1, 0, 0), 2, False)
        probe = Probe()
        oracle = ExpertOracle(probe, graph, {0, 1, 40}, 1, max_nodes=10, max_discoveries=0)
        self.assertEqual(oracle.label((40, 0, 0), Position(40, 0, 0, False)), 0)
        self.assertEqual(oracle.recovery_labels, 1)
        self.assertGreater(probe.calls, 1)

    def test_incomplete_or_failed_expert_is_not_training_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'demo'
            rec = DemonstrationRecorder(FakeEnv(), path, LOOKUPS, 'greedy_expert')
            rec.reset()
            rec.step(0)
            with self.assertRaisesRegex(ValueError, 'still recording'):
                DemonstrationDataset([path])
            rec.finish('no_known_path')
            with self.assertRaisesRegex(ValueError, 'must have reached'):
                DemonstrationDataset([path])
            with self.assertRaisesRegex(RuntimeError, 'finalized'):
                rec.step(0)
            rec.close()

    def test_resolve_worst_curriculum_snapshot_with_checksum(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'worst.state').write_bytes(b'data')
            (root / 'current.json').write_text(json.dumps({'states': {'worst': {
                'filename': 'worst.state', 'sha256': hashlib.sha256(b'data').hexdigest()}}}))
            self.assertEqual(resolve_start('unused', root), root / 'worst.state')
            (root / 'worst.state').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                resolve_start('unused', root)

    def test_clone_updates_policy_without_using_ppo_optimizer(self):
        import torch
        from stable_baselines3 import PPO
        from pokemon_game.demonstrations.behavioral_cloning import clone
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            path = session(Path(directory) / 'demo')
            dataset = DemonstrationDataset([path])
            model = PPO('MultiInputPolicy', FakeEnv(), n_steps=2, batch_size=2,
                        n_epochs=1, seed=0, device='cpu', verbose=0)
            before = model.policy.action_net.weight.detach().clone()
            losses = clone(model, dataset, epochs=3, batch_size=2, learning_rate=0.001)
            self.assertEqual(len(losses), 3)
            self.assertTrue(all(np.isfinite(loss) for loss in losses))
            self.assertFalse(torch.equal(before, model.policy.action_net.weight.detach()))
            self.assertEqual(len(model.policy.optimizer.state), 0)


class ScheduleTests(unittest.TestCase):
    def test_fixed_evaluations_match_for_control_and_imitation(self):
        from pokemon_game.demonstrations.periodic_training import run_schedule

        class FakeModel:
            n_steps, n_envs, num_timesteps = 2, 2, 0
            def learn(self, *, total_timesteps, reset_num_timesteps):
                self.num_timesteps += total_timesteps
            def save(self, path): pass

        config = {'imitation_learning': {'enabled': False,
            'initial_pretraining': {'enabled': False, 'epochs': 1},
            'periodic_training': {'enabled': True, 'epochs': 1, 'interval_rollouts': 2},
            'batch_size': 4, 'learning_rate': 0.001}}
        evaluations = []
        with tempfile.TemporaryDirectory() as directory:
            run_schedule(FakeModel(), None, None, config, 5, Path(directory),
                         evaluator=lambda model, rollout, phase: evaluations.append((rollout, phase)))
        self.assertEqual(evaluations, [(0, 'before_ppo'), (2, 'after_update'),
                                       (4, 'after_update'), (5, 'after_update')])

    def test_rollout_progress_tracks_original_reward_and_episode_boundaries(self):
        from pokemon_game.demonstrations.periodic_training import TrainingProgress
        import time

        class Model:
            n_steps, n_envs, num_timesteps = 2, 2, 100

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = Model()
            callback = TrainingProgress(root, total_rollouts=2, started=time.perf_counter(),
                                        initial_steps=100, goal_enabled=True)
            callback.model = model
            callback.locals = {'rewards': [8.0, 2.0], 'dones': [False, False],
                               'infos': [{'original_reward': 1.0}, {}]}
            model.num_timesteps = 102
            callback.on_step()
            callback.locals = {'rewards': [5.0, 3.0], 'dones': [True, False],
                               'infos': [{'original_reward': 0.5, 'goal_reached': True}, {}]}
            model.num_timesteps = 104
            callback.on_step()
            callback.on_rollout_end()
            first = json.loads((root / 'progress.jsonl').read_text().splitlines()[0])
            self.assertEqual((first['rollout'], first['transitions_this_run']), (1, 4))
            self.assertEqual(first['mean_original_return'], 1.5)
            self.assertEqual(first['mean_episode_length'], 2)
            self.assertEqual(first['goal_successes'], 1)
            # The other worker's unfinished episode continues into the next chunk.
            callback.locals = {'rewards': [0.0, 4.0], 'dones': [False, True], 'infos': [{}, {}]}
            model.num_timesteps = 106
            callback.on_step()
            callback.locals = {'rewards': [0.0, 0.0], 'dones': [False, False], 'infos': [{}, {}]}
            model.num_timesteps = 108
            callback.on_step()
            callback.on_rollout_end()
            second = json.loads((root / 'progress.jsonl').read_text().splitlines()[1])
            self.assertEqual(second['rollout'], 2)
            self.assertEqual(second['completed_episodes'], 2)
            self.assertEqual(second['mean_original_return'], (1.5 + 9.0) / 2)

    def test_imitation_occurs_only_between_complete_ppo_chunks(self):
        from pokemon_game.demonstrations.periodic_training import run_schedule

        class FakeModel:
            n_steps, n_envs, num_timesteps = 2, 2, 0
            def learn(self, *, total_timesteps, reset_num_timesteps):
                self.num_timesteps += total_timesteps
            def save(self, path):
                pass

        class Dataset:
            sessions = [{'path': 'synthetic'}]
            def __len__(self): return 3

        config = {'imitation_learning': {'enabled': True,
            'initial_pretraining': {'enabled': True, 'epochs': 2},
            'periodic_training': {'enabled': True, 'epochs': 1, 'interval_rollouts': 2},
            'batch_size': 4, 'learning_rate': 0.001}}
        with tempfile.TemporaryDirectory() as directory, \
             patch('pokemon_game.demonstrations.periodic_training.accuracy', return_value=0.5), \
             patch('pokemon_game.demonstrations.periodic_training.clone', return_value=[1.0]):
            root = Path(directory)
            result = run_schedule(FakeModel(), Dataset(), None, config, 5, root)
            events = [json.loads(x) for x in (root / 'schedule.jsonl').read_text().splitlines()]
            self.assertEqual([(e['phase'], e['timesteps']) for e in events], [
                ('initial_pretraining', 0), ('ppo_update_complete', 8), ('periodic_imitation', 8),
                ('ppo_update_complete', 16), ('periodic_imitation', 16), ('ppo_update_complete', 20)])
            self.assertEqual(result['imitation_updates'], 3)

    def test_reloads_demonstrations_before_each_imitation_update(self):
        from pokemon_game.demonstrations.periodic_training import run_schedule

        class FakeModel:
            n_steps, n_envs, num_timesteps = 2, 1, 0
            def learn(self, *, total_timesteps, reset_num_timesteps):
                self.num_timesteps += total_timesteps
            def save(self, path):
                pass

        class Dataset:
            def __init__(self, version): self.sessions = [{'version': version}]
            def __len__(self): return 1

        versions = []
        def loader():
            versions.append(len(versions) + 1)
            return Dataset(versions[-1])
        config = {'imitation_learning': {'enabled': True,
            'initial_pretraining': {'enabled': True, 'epochs': 1},
            'periodic_training': {'enabled': True, 'epochs': 1, 'interval_rollouts': 1},
            'batch_size': 4, 'learning_rate': 0.001}}
        with tempfile.TemporaryDirectory() as directory, \
             patch('pokemon_game.demonstrations.periodic_training.accuracy', return_value=0.5), \
             patch('pokemon_game.demonstrations.periodic_training.clone', return_value=[1.0]) as cloning:
            run_schedule(FakeModel(), Dataset(0), None, config, 3, Path(directory), dataset_loader=loader)
            self.assertEqual(versions, [1, 2, 3])
            self.assertEqual([call.args[1].sessions[0]['version'] for call in cloning.call_args_list], versions)


if __name__ == '__main__':
    unittest.main()
