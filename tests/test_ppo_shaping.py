"""Behavioral tests for discounted shaping and episode boundaries."""
import unittest

import gymnasium as gym

from pokemon_game.reward_shaping import MapPotential, PotentialShaping


class WalkEnv(gym.Env):
    def __init__(self):
        self.pyboy = type('Boy', (), {'memory': bytearray(65536)})()
        self.start = 0

    def reset(self, **kwargs):
        self.pyboy.memory[0xD362] = self.start
        return 0, {}

    def step(self, action):
        position, terminated, truncated = action
        self.pyboy.memory[0xD362] = position
        return 0, 3.0, terminated, truncated, {}


def potential(key):
    values = {0: 0.2, 1: 0.5, 2: 0.8}
    return (values[key[1]], False) if key[1] in values else (0.0, True)


class PPOShapingTests(unittest.TestCase):
    def make(self, beta=1):
        return PotentialShaping(WalkEnv(), potential, beta, 0.9)

    def test_zero_beta_leaves_rewards_unchanged(self):
        env = self.make(beta=0)
        env.reset()
        _, reward, _, _, info = env.step((1, False, False))
        self.assertEqual(reward, 3.0)
        self.assertEqual(info['original_reward'], 3.0)
        self.assertEqual(info['navigation_shaping'], 0.0)

    def test_discounted_movement_and_reset(self):
        env = self.make()
        env.reset()
        _, _, _, _, info = env.step((1, False, False))
        self.assertAlmostEqual(info['navigation_shaping'], 0.9 * 0.5 - 0.2)
        self.assertGreater(info['navigation_shaping'], 0)
        _, _, _, _, info = env.step((0, False, False))
        self.assertAlmostEqual(info['navigation_shaping'], 0.9 * 0.2 - 0.5)
        self.assertLess(info['navigation_shaping'], 0)
        env.step((2, False, False))
        env.reset()
        _, _, _, _, info = env.step((1, False, False))
        self.assertAlmostEqual(info['navigation_shaping'], 0.9 * 0.5 - 0.2)

    def test_increasing_potential_can_have_negative_discounted_shaping(self):
        values = {0: 0.8, 1: 0.81}
        env = PotentialShaping(WalkEnv(), lambda key: (values[key[1]], False), 1.0, 0.9)
        env.reset()
        _, _, _, _, info = env.step((1, False, False))
        self.assertAlmostEqual(info['navigation_shaping'], 0.9 * 0.81 - 0.8)
        self.assertLess(info['navigation_shaping'], 0)

    def test_goal_termination_precedes_shaping(self):
        from pokemon_game.ppo_navigation_experiment import GoalEpisode

        class GoalEnv(WalkEnv):
            def step(self, action):
                self.pyboy.memory[0xD35E] = 1
                return 0, 3.0, False, True, {}

        env = PotentialShaping(GoalEpisode(GoalEnv(), 1), potential, 1.0, 0.9)
        env.reset()
        _, reward, terminated, truncated, info = env.step(None)
        self.assertTrue(info['goal_reached'])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, 3.0 - 0.2)

    def test_termination_and_truncation_are_distinct(self):
        env = self.make()
        env.reset()
        _, _, terminated, truncated, info = env.step((2, True, False))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(info['navigation_shaping'], -0.2)
        env.reset()
        _, _, terminated, truncated, info = env.step((2, False, True))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(info['navigation_shaping'], 0.9 * 0.8 - 0.2)

    def test_missing_states_report_fallback_and_workers_are_independent(self):
        one, two = self.make(), self.make()
        one.reset()
        two.reset()
        _, _, _, _, info = one.step((9, False, False))
        self.assertTrue(info['navigation_potential_missing'])
        self.assertAlmostEqual(info['navigation_shaping'], -0.2)
        _, _, _, _, info = two.step((1, False, False))
        self.assertFalse(info['navigation_potential_missing'])
        self.assertAlmostEqual(info['navigation_shaping'], 0.9 * 0.5 - 0.2)
        self.assertEqual(MapPotential({0: 0.25})((1, 0, 0)), (0.0, True))


if __name__ == '__main__':
    unittest.main()
