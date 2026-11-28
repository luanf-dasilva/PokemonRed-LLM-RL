"""Tests of battle semantics, including a local-ROM integration when available."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pokemon_game.battle_env import (BATTLE_FLAG, BATTLE_TYPE, MENU_X, MENU_Y, TILEMAP,
                                      PokemonBattleEnv, battle_menu, mon, validate_decision)
from pokemon_game.capture_battles import capture
from pokemon_game.evaluate_greedy import prepare_start
from pokemon_game.state_encoder import load_lookups

ROOT = Path(__file__).resolve().parent.parent


class MemoryTests(unittest.TestCase):
    def test_command_menu_requires_wild_battle_and_menu_geometry(self):
        memory = bytearray(65536)
        memory[BATTLE_FLAG] = 1
        memory[BATTLE_TYPE] = 0
        memory[MENU_Y], memory[MENU_X] = 14, 9
        for i, c in enumerate('FIGHT'):
            memory[TILEMAP + 14 * 20 + 10 + i] = 0x80 + ord(c) - ord('A')
        for i, c in enumerate('RUN'):
            memory[TILEMAP + 16 * 20 + 16 + i] = 0x80 + ord(c) - ord('A')
        self.assertTrue(battle_menu(memory))
        memory[BATTLE_FLAG] = 2
        self.assertFalse(battle_menu(memory))
        memory[BATTLE_FLAG] = 1
        memory[MENU_Y] = 12
        self.assertFalse(battle_menu(memory))
        with self.assertRaises(ValueError):
            validate_decision(memory)

    def test_mon_fields_use_big_endian_and_mask_pp_ups(self):
        memory = bytearray(65536)
        base = 0xD014
        memory[base] = 177
        memory[base + 1], memory[base + 2] = 1, 2
        memory[base + 14] = 6
        memory[base + 15], memory[base + 16] = 2, 0
        memory[base + 8] = 33
        memory[base + 25] = 0xC5
        value = mon(memory, base)
        self.assertEqual((value['species_id'], value['hp'], value['max_hp']), (177, 258, 512))
        self.assertEqual((value['moves'][0], value['pp'][0]), (33, 5))


@unittest.skipUnless((ROOT / 'PokemonRed.gb').exists() and
                     (ROOT / 'pokemon_rag/data/pret-pokered/processed/lookups.json').exists(),
                     'Local ROM and generated lookups required')
class BattleIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.rom = ROOT / 'PokemonRed.gb'
        start = cls.root / 'pallet.state'
        lookups = load_lookups(ROOT / 'pokemon_rag/data/pret-pokered/processed/lookups.json')
        prepare_start(cls.rom, ROOT / 'has_pokedex.state', start, lookups)
        capture(cls.rom, start, ROOT / 'research/navigation_experiments/pallet-to-viridian/outcomes.jsonl',
                cls.root / 'archive', 'ppo', 0, 0, 1)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.env = PokemonBattleEnv(self.rom, self.root / 'archive')

    def tearDown(self):
        self.env.close()

    def test_capture_reaches_command_menu_and_attack_wins(self):
        obs, info = self.env.reset(options={'scenario': 0})
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertEqual(info['action_mask'].tolist(), [True, True, False, False, True])
        outcomes, frames = [], 0
        for _ in range(10):
            obs, _, terminated, truncated, info = self.env.step(0)
            outcomes.append(info['outcome'])
            frames += info['emulator_frames']
            self.assertTrue(self.env.observation_space.contains(obs))
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(outcomes[-1], 'victory')
        self.assertGreater(frames, len(outcomes) * 24)
        with self.assertRaises(RuntimeError):
            self.env.step(0)

    def test_invalid_move_and_escape(self):
        self.env.reset(options={'scenario': 0})
        self.env.pyboy.memory[0xD06D] = 0x11  # First move disabled for one turn.
        self.assertFalse(self.env.action_masks()[0])
        self.env.pyboy.memory[0xD06D] = 0
        _, reward, terminated, truncated, info = self.env.step(3)
        self.assertLess(reward, 0)
        self.assertFalse(terminated or truncated)
        self.assertEqual(info['emulator_frames'], 0)
        _, _, terminated, truncated, info = self.env.step(4)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['outcome'], 'escaped')

    def test_move_two_uses_requested_pp_and_next_decision(self):
        self.env.reset(options={'scenario': 0})
        before = mon(self.env.pyboy.memory, 0xD014)
        _, _, terminated, truncated, info = self.env.step(1)
        after = mon(self.env.pyboy.memory, 0xD014)
        self.assertFalse(terminated or truncated)
        self.assertEqual(info['outcome'], 'ongoing')
        self.assertEqual(after['pp'][0], before['pp'][0])
        self.assertEqual(after['pp'][1], before['pp'][1] - 1)

    def test_snapshot_integrity_checked(self):
        path = self.root / 'archive' / 'battle_0002.state'
        old = path.read_bytes()
        try:
            path.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                self.env.reset(options={'scenario': 1})
        finally:
            path.write_bytes(old)


if __name__ == '__main__':
    unittest.main()
