import json
import unittest

from pokemon_game.emulator_simulator import EmulatorSimulator, Position
from pokemon_game.greedy_policy import select_action, tile_value
from pokemon_game.tile_graph import discover, distances_to_target, manhattan_distance, position_key
from pokemon_game.value_function import HierarchicalNavigationValue


class FakePyBoy:
    def __init__(self, transitions):
        self.memory = bytearray(65536)
        self.transitions = transitions
        self.button = None
        self.moved = False

    def button_press(self, button):
        self.button, self.moved = button, False

    def button_release(self, button):
        self.button = None

    def tick(self, count, render):
        if self.button is not None and not self.moved:
            key = (self.memory[0xD35E], self.memory[0xD362], self.memory[0xD361], self.button)
            target = self.transitions.get(key)
            if target is not None:
                self.memory[0xD35E], self.memory[0xD362], self.memory[0xD361], self.memory[0xD057] = target
            self.moved = True

    def save_state(self, stream):
        stream.write(json.dumps([self.memory[0xD35E], self.memory[0xD362],
                                 self.memory[0xD361], self.memory[0xD057]]).encode())

    def load_state(self, stream):
        self.memory[0xD35E], self.memory[0xD362], self.memory[0xD361], self.memory[0xD057] = json.loads(stream.read())
        self.button = None


class TileGraphTests(unittest.TestCase):
    def setUp(self):
        self.transitions = {(0, 0, 0, "right"): (0, 1, 0, 0),
                            (0, 1, 0, "right"): (1, 0, 0, 0),
                            (0, 0, 0, "up"): (0, 0, 1, 1)}
        self.sim = EmulatorSimulator(FakePyBoy(self.transitions))

    def test_discovery_is_directed_and_restores_original(self):
        root = self.sim.snapshot()
        graph = discover(self.sim, target_map_id=1, allowed_maps={0, 1})
        self.assertEqual(graph.target, (1, 0, 0))
        self.assertEqual(self.sim.snapshot(), root)
        self.assertEqual(distances_to_target(graph)[graph.start], 2)
        self.assertNotIn((0, 0, 1), graph.edges)
        self.assertEqual(self.sim.committed_actions, 0)
        self.assertGreater(self.sim.simulated_actions, 0)

    def test_greedy_probes_from_same_root_and_restores(self):
        graph = discover(self.sim, target_map_id=1, allowed_maps={0, 1})
        distances = distances_to_target(graph)
        root = self.sim.snapshot()
        action = select_action(self.sim, lambda after: tile_value(distances, after))
        self.assertEqual(action, 2)
        self.assertEqual(self.sim.snapshot(), root)
        self.assertEqual(position_key(self.sim.step(action)), (0, 1, 0))
        self.assertEqual(manhattan_distance((1, 4), (3, 5)), 3)

    def test_restores_after_evaluation_error(self):
        root = self.sim.snapshot()
        def fail(_):
            raise ValueError("broken evaluator")
        with self.assertRaisesRegex(ValueError, "broken evaluator"):
            select_action(self.sim, fail)
        self.assertEqual(self.sim.snapshot(), root)

    def test_hierarchical_value_improves_toward_observed_exit(self):
        graph = discover(self.sim, target_map_id=1, allowed_maps={0, 1})
        value = HierarchicalNavigationValue(graph, local_weight=0.5)
        start = value.evaluate(Position(0, 0, 0, False))
        exit_tile = value.evaluate(Position(0, 1, 0, False))
        target = value.evaluate(Position(1, 0, 0, False))
        self.assertLess(start, exit_tile)
        self.assertLess(exit_tile, target)
        self.assertIsNone(value.evaluate(Position(0, 9, 9, False)))
        with self.assertRaises(ValueError):
            HierarchicalNavigationValue(graph, local_weight=1)


if __name__ == "__main__":
    unittest.main()
