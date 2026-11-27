import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from pokemon_game.emulator_simulator import Position
from pokemon_game.goal_resolver import resolve_goal
from pokemon_game.greedy_policy import correct_transition, select_model_action
from pokemon_game.tile_graph import Discovery, distances_to_target, load_graph, save_graph


class CachedNavigationTests(unittest.TestCase):
    def setUp(self):
        self.start, self.middle, self.target = (0, 0, 0), (0, 1, 0), (1, 0, 0)
        self.graph = Discovery({self.start: {2: self.middle}, self.middle: {2: self.target}},
                               self.start, self.target, 3, False)

    def test_round_trip_and_context_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            save_graph(path, self.graph, {"rom": "a"})
            loaded = load_graph(path, {"rom": "a"})
            self.assertEqual(loaded, self.graph)
            with self.assertRaises(ValueError):
                load_graph(path, {"rom": "b"})
            data = json.loads(path.read_text())
            data["edges"][0]["actions"]["7"] = [0, 0, 0]
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_graph(path, {"rom": "a"})

    def test_model_chooses_directed_path_and_replans_after_mismatch(self):
        distances = distances_to_target(self.graph)
        self.assertEqual(select_model_action(self.graph, Position(*self.start, False), distances), 2)
        changed = correct_transition(self.graph, self.start, 2, self.middle,
                                     Position(*self.start, False), {0, 1})
        self.assertTrue(changed)
        self.assertNotIn(self.start, distances_to_target(self.graph))
        with self.assertRaises(RuntimeError):
            select_model_action(self.graph, Position(*self.start, False), distances_to_target(self.graph))

    def test_battle_never_creates_a_navigation_edge(self):
        self.assertTrue(correct_transition(self.graph, self.start, 2, self.middle,
                                           Position(*self.middle, True), {0, 1}))
        self.assertNotIn(2, self.graph.edges[self.start])

    def test_goal_resolver_rejects_unsupported_and_unknown_goals(self):
        lookups = {"maps": {"VIRIDIAN_CITY": {"id": 1}}}
        self.assertEqual(resolve_goal({"goal_type": "REACH_LOCATION", "target": "Viridian City"}, lookups), 1)
        for goal in ({"goal_type": "HEAL", "target": "Viridian City"},
                     {"goal_type": "REACH_LOCATION", "target": "Unknown"}):
            with self.assertRaises(ValueError):
                resolve_goal(goal, lookups)

    def test_shaping_uses_terminal_zero_and_bootstraps_truncation(self):
        import gymnasium as gym
        from pokemon_game.reward_shaping import NavigationShaping

        class FakeEnv(gym.Env):
            def __init__(self):
                self.pyboy = type("FakePyBoy", (), {"memory": bytearray(65536)})()
                self.terminal = False
            def reset(self, **kwargs):
                self.pyboy.memory[0xD362] = 0
                return None, {}
            def step(self, action):
                self.pyboy.memory[0xD362] = 1
                return None, 2.0, self.terminal, not self.terminal, {}

        with tempfile.TemporaryDirectory() as directory:
            rom, path = Path(directory) / "rom", Path(directory) / "graph"
            rom.write_bytes(b"rom")
            context = {"rom_sha256": hashlib.sha256(b"rom").hexdigest(), "target_map": 1}
            save_graph(path, self.graph, context)
            env = NavigationShaping(FakeEnv(), path, rom, beta=2.0, gamma=0.9)
            env.reset()
            _, reward, _, truncated, info = env.step(0)
            self.assertTrue(truncated)
            self.assertAlmostEqual(reward, 2 + 2 * (0.9 / 2 - 1 / 3))
            self.assertAlmostEqual(info["navigation_shaping"], reward - 2)
            env.reset()
            env.unwrapped.terminal = True
            _, reward, terminated, _, _ = env.step(0)
            self.assertTrue(terminated)
            self.assertAlmostEqual(reward, 2 - 2 / 3)


if __name__ == "__main__":
    unittest.main()
