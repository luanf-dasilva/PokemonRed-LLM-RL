"""One-on-one wild battles using semantic move and escape actions.

Addresses and tile-menu layouts correspond to the English Gen I ROM and the
pret/pokered disassembly. This does not support trainer, Safari, item, switch,
learning-move, multi-party or capture workflows.
"""
from enum import IntEnum
import hashlib
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pokemon_game.emulator_simulator import BATTLE_FLAG
from pokemon_game.state_encoder import MAP_ID, PARTY_COUNT, load_lookups

ROOT = Path(__file__).resolve().parent.parent
TILEMAP = 0xC3A0
ENEMY, PLAYER = 0xCFE5, 0xD014
MENU_Y, MENU_X, MENU_ITEM = 0xCC24, 0xCC25, 0xCC26
BATTLE_TYPE = 0xD05A
DISABLED_MOVE = 0xD06D


class BattleAction(IntEnum):
    MOVE_1 = 0
    MOVE_2 = 1
    MOVE_3 = 2
    MOVE_4 = 3
    RUN = 4


def word(memory, address):
    return (memory[address] << 8) | memory[address + 1]


def mon(memory, address):
    return {'species_id': memory[address], 'hp': word(memory, address + 1),
            'status': memory[address + 4], 'type_ids': [memory[address + 5], memory[address + 6]],
            'moves': [memory[address + 8 + i] for i in range(4)],
            'level': memory[address + 14], 'max_hp': word(memory, address + 15),
            'attack': word(memory, address + 17), 'defense': word(memory, address + 19),
            'speed': word(memory, address + 21), 'special': word(memory, address + 23),
            'pp': [memory[address + 25 + i] & 0x3F for i in range(4)]}


def _text(memory, x, y, length):
    return bytes(memory[TILEMAP + y * 20 + x + i] for i in range(length))


def _encoded(text):
    return bytes(0x80 + ord(c) - ord('A') for c in text)


def battle_menu(memory):
    """Detect the normal battle command screen, not a battle-entry animation."""
    return (memory[BATTLE_FLAG] == 1 and memory[BATTLE_TYPE] == 0 and
            _text(memory, 10, 14, 5) == _encoded('FIGHT') and
            _text(memory, 16, 16, 3) == _encoded('RUN') and
            memory[MENU_Y] == 14 and memory[MENU_X] in (9, 15))


def command_text_visible(memory):
    """Stop sending dialogue input as soon as the command box starts drawing."""
    return memory[BATTLE_FLAG] == 1 and _text(memory, 10, 14, 5) == _encoded('FIGHT')


def move_menu(memory):
    return (memory[BATTLE_FLAG] == 1 and memory[MENU_Y] == 12 and memory[MENU_X] == 5 and
            _text(memory, 1, 9, 4) == _encoded('TYPE'))


def validate_decision(memory):
    if not battle_menu(memory) or memory[PARTY_COUNT] != 1:
        raise ValueError('Expected a wild one-party battle at the command menu')
    ours, theirs = mon(memory, PLAYER), mon(memory, ENEMY)
    for name, creature in (('player', ours), ('opponent', theirs)):
        if not creature['species_id'] or not 0 < creature['hp'] <= creature['max_hp']:
            raise ValueError(f'Invalid {name} battle data')
    if not any(move and pp for move, pp in zip(ours['moves'], ours['pp'])):
        raise ValueError('No supported move has remaining PP')
    return ours, theirs


def describe(memory, lookups):
    ours, theirs = validate_decision(memory)
    maps = {entry['id']: name for name, entry in lookups['maps'].items()}
    species = {entry['id']: name for name, entry in lookups['species'].items()}
    for creature in (ours, theirs):
        creature['species'] = species.get(creature['species_id'], f"UNKNOWN_{creature['species_id']:02X}")
        creature['hp_fraction'] = creature['hp'] / creature['max_hp']
    return {'battle_type': 'wild', 'map_id': memory[MAP_ID],
            'location': maps.get(memory[MAP_ID], f"UNKNOWN_MAP_{memory[MAP_ID]:02X}"),
            'player': ours, 'opponent': theirs}


class BattleController:
    """Bounded button/timing driver with frame and controller-input accounting."""
    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.frames = 0
        self.inputs = 0

    def tick(self, frames):
        self.pyboy.tick(frames, True)
        self.frames += frames

    def press(self, button):
        self.pyboy.button_press(button)
        self.tick(8)
        self.pyboy.button_release(button)
        self.tick(16)
        self.inputs += 1

    def advance_text(self):
        # A long held press can leak into the next command menu and select an
        # unintended move. Release immediately and allow the screen to settle.
        self.pyboy.button_press('a')
        self.tick(1)
        self.pyboy.button_release('a')
        self.tick(31)
        self.inputs += 1

    def advance(self, max_frames=6000, require_departure=False):
        """Advance dialogue until a new menu or a battle exit, with a hard cap."""
        start = self.frames
        departed = not require_departure
        while self.frames - start < max_frames:
            flag = self.pyboy.memory[BATTLE_FLAG]
            if flag != 1:
                return 'ended'
            menu = battle_menu(self.pyboy.memory)
            if not menu:
                departed = True
            if menu and departed:
                # Wait briefly to make sure menu setup has reached input polling.
                self.tick(8)
                if battle_menu(self.pyboy.memory):
                    return 'decision'
            if command_text_visible(self.pyboy.memory):
                self.tick(4)
                continue
            self.tick(32)
            if self.pyboy.memory[BATTLE_FLAG] == 1 and not command_text_visible(self.pyboy.memory):
                self.advance_text()
        return 'timeout'

    def command(self, action):
        if not battle_menu(self.pyboy.memory):
            raise RuntimeError('Not at a battle decision')
        target_col, target_row = (15, 1) if action == BattleAction.RUN else (9, 0)
        if self.pyboy.memory[MENU_X] != target_col:
            self.press('right' if target_col == 15 else 'left')
        if self.pyboy.memory[MENU_ITEM] != target_row:
            self.press('down' if target_row else 'up')
        if not battle_menu(self.pyboy.memory) or self.pyboy.memory[MENU_X] != target_col or self.pyboy.memory[MENU_ITEM] != target_row:
            raise RuntimeError('Failed to select battle command')
        self.press('a')
        if action != BattleAction.RUN:
            for _ in range(30):
                if move_menu(self.pyboy.memory):
                    break
                self.tick(8)
            else:
                raise RuntimeError('Move menu did not appear')
            selected = int(action) + 1  # Move menu uses 1-based indices.
            for _ in range(8):
                current = self.pyboy.memory[MENU_ITEM]
                if current == selected:
                    break
                self.press('down' if current < selected else 'up')
            if not move_menu(self.pyboy.memory) or self.pyboy.memory[MENU_ITEM] != selected:
                raise RuntimeError('Failed to select move')
            self.press('a')
        return self.advance(require_departure=True)


class PokemonBattleEnv(gym.Env):
    """A single-action-space Gymnasium battle environment over archived states."""
    metadata = {'render_modes': []}

    def __init__(self, rom, archive, max_actions=30, invalid_penalty=-0.05):
        from pyboy import PyBoy
        self.rom = Path(rom).resolve()
        self.archive = Path(archive).resolve()
        metadata = json.loads((self.archive / 'manifest.json').read_text())
        if metadata.get('version') != 1:
            raise ValueError('Unsupported battle archive version')
        if metadata['rom_sha256'] != hashlib.sha256(self.rom.read_bytes()).hexdigest():
            raise ValueError('Battle archive ROM mismatch')
        self.scenarios = metadata['scenarios']
        if not self.scenarios or max_actions < 1:
            raise ValueError('Archive must contain scenarios and max_actions must be positive')
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(30,), dtype=np.float32)
        self.max_actions, self.invalid_penalty = max_actions, invalid_penalty
        self.pyboy = PyBoy(str(self.rom), window='null')
        self.pyboy.set_emulation_speed(0)
        self.controller = BattleController(self.pyboy)
        self.ended = True

    def _observation(self):
        ours, theirs = mon(self.pyboy.memory, PLAYER), mon(self.pyboy.memory, ENEMY)
        values = [ours['hp'] / max(ours['max_hp'], 1), theirs['hp'] / max(theirs['max_hp'], 1),
                  ours['level'] / 100, theirs['level'] / 100, ours['status'] / 255, theirs['status'] / 255,
                  ours['species_id'] / 255, theirs['species_id'] / 255]
        values += [v / 255 for v in ours['type_ids'] + theirs['type_ids']]
        values += [v / 255 for v in ours['moves']]
        values += [v / 63 for v in ours['pp']]
        values += [min(ours[key], 999) / 999 for key in ('attack', 'defense', 'speed', 'special')]
        values += [min(theirs[key], 999) / 999 for key in ('attack', 'defense', 'speed', 'special')]
        values += [(self.pyboy.memory[DISABLED_MOVE] >> 4) / 15]
        values += [self.actions / self.max_actions]
        return np.clip(np.array(values, dtype=np.float32), 0, 1)

    def action_masks(self):
        ours = mon(self.pyboy.memory, PLAYER)
        disabled = self.pyboy.memory[DISABLED_MOVE] >> 4
        return np.array([bool(move and pp and i + 1 != disabled)
                         for i, (move, pp) in enumerate(zip(ours['moves'], ours['pp']))] + [True])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = int(self.np_random.integers(len(self.scenarios))) if options is None else int(options.get('scenario', 0))
        if not 0 <= index < len(self.scenarios):
            raise ValueError('Invalid scenario index')
        entry = self.scenarios[index]
        filename = entry['snapshot']
        if Path(filename).name != filename:
            raise ValueError('Invalid snapshot filename')
        data = (self.archive / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['snapshot_sha256']:
            raise ValueError('Battle snapshot hash mismatch')
        from io import BytesIO
        self.pyboy.load_state(BytesIO(data))
        validate_decision(self.pyboy.memory)
        self.controller.frames = self.controller.inputs = 0
        self.actions, self.ended = 0, False
        return self._observation(), {'scenario': index, 'action_mask': self.action_masks()}

    def step(self, action):
        if self.ended:
            raise RuntimeError('Reset required after episode end')
        if not self.action_space.contains(action):
            raise ValueError('Invalid abstract battle action')
        self.actions += 1
        frames, inputs = self.controller.frames, self.controller.inputs
        if not self.action_masks()[int(action)]:
            truncated = self.actions >= self.max_actions
            self.ended = truncated
            return self._observation(), self.invalid_penalty, False, truncated, {
                'outcome': 'invalid_move', 'emulator_frames': 0, 'controller_inputs': 0,
                'action_mask': self.action_masks()}
        before = mon(self.pyboy.memory, PLAYER), mon(self.pyboy.memory, ENEMY)
        try:
            result = self.controller.command(BattleAction(int(action)))
        except RuntimeError as exc:
            self.ended = True
            return self._observation(), 0.0, False, True, {'outcome': 'controller_error', 'error': str(exc),
                'emulator_frames': self.controller.frames - frames, 'controller_inputs': self.controller.inputs - inputs}
        after = mon(self.pyboy.memory, PLAYER), mon(self.pyboy.memory, ENEMY)
        delta_enemy = (before[1]['hp'] - after[1]['hp']) / max(before[1]['max_hp'], 1)
        delta_player = (before[0]['hp'] - after[0]['hp']) / max(before[0]['max_hp'], 1)
        outcome = 'ongoing'
        if result == 'ended':
            if after[0]['hp'] == 0 or self.pyboy.memory[BATTLE_FLAG] == 255:
                outcome = 'defeat'
            elif after[1]['hp'] == 0:
                outcome = 'victory'
            elif action == BattleAction.RUN:
                outcome = 'escaped'
            else:
                outcome = 'ended_unknown'
        terminated = result == 'ended'
        truncated = result == 'timeout' or (not terminated and self.actions >= self.max_actions)
        if result == 'timeout':
            outcome = 'controller_timeout'
        self.ended = terminated or truncated
        reward = delta_enemy - delta_player + (1 if outcome == 'victory' else -1 if outcome == 'defeat' else 0)
        info = {'outcome': outcome, 'enemy_hp_delta': delta_enemy, 'player_hp_delta': delta_player,
                'emulator_frames': self.controller.frames - frames, 'controller_inputs': self.controller.inputs - inputs,
                'total_emulator_frames': self.controller.frames, 'abstract_actions': self.actions}
        if not self.ended:
            info['action_mask'] = self.action_masks()
        return self._observation(), float(reward), terminated, truncated, info

    def close(self):
        self.pyboy.stop(save=False)
