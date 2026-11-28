"""Capture stable wild battle command menus from a recorded exploration trace."""
import argparse
import hashlib
import io
import json
from pathlib import Path

from pyboy import PyBoy

from pokemon_game.battle_env import BattleController, PokemonBattleEnv, describe
from pokemon_game.emulator_simulator import BATTLE_FLAG
from pokemon_game.state_encoder import load_lookups

ROOT = Path(__file__).resolve().parent.parent
BUTTONS = ('down', 'left', 'right', 'up', 'a', 'b', 'start')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def capture(rom, start, trace, output, policy, seed, wait_frames, derive_moves, wait_before_action=1):
    rom, start, trace, output = map(lambda p: Path(p).resolve(), (rom, start, trace, output))
    if output.exists():
        raise FileExistsError(f'Choose a new archive directory: {output}')
    if wait_frames < 0 or derive_moves < 0 or wait_before_action < 1:
        raise ValueError('Frame wait and derived count must be nonnegative')
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row.get('policy') == policy and row.get('seed') == seed]
    if len(selected) != 1:
        raise ValueError('Trace must contain exactly one matching policy and seed')
    actions = selected[0]['actions']
    if any(type(action) is not int or action not in range(7) for action in actions):
        raise ValueError('Invalid exploration actions')
    if wait_frames and wait_before_action > len(actions):
        raise ValueError('Wait action index is outside the trace')
    lookups = load_lookups(ROOT / 'pokemon_rag/data/pret-pokered/processed/lookups.json')
    pyboy = PyBoy(str(rom), window='null')
    pyboy.set_emulation_speed(0)
    try:
        with start.open('rb') as stream:
            pyboy.load_state(stream)
        driver = BattleController(pyboy)
        index = None
        for index, action in enumerate(actions, 1):
            if wait_frames and index == wait_before_action:
                driver.tick(wait_frames)
            driver.press(BUTTONS[action])
            if pyboy.memory[BATTLE_FLAG]:
                break
        else:
            raise RuntimeError('Trace did not encounter a battle')
        if pyboy.memory[BATTLE_FLAG] != 1:
            raise ValueError('Encounter is not a wild battle')
        result = driver.advance()
        if result != 'decision':
            raise RuntimeError(f'Could not reach a battle decision: {result}')
        data = io.BytesIO()
        pyboy.save_state(data)
        snapshot = data.getvalue()
        description = describe(pyboy.memory, lookups)
        output.mkdir(parents=True)
        (output / 'battle_0001.state').write_bytes(snapshot)
        entry = {'snapshot': 'battle_0001.state', 'snapshot_sha256': digest(snapshot), **description,
                 'provenance': {'kind': 'recorded_exploration_trace', 'trace_sha256': digest(trace.read_bytes()),
                    'policy': policy, 'seed': seed, 'start_sha256': digest(start.read_bytes()),
                    'wait_frames': wait_frames, 'wait_before_action': wait_before_action, 'exploration_actions': index,
                    'capture_emulator_frames': driver.frames, 'capture_controller_inputs': driver.inputs}}
        manifest = {'version': 1, 'rom_sha256': digest(rom.read_bytes()), 'scenarios': [entry]}
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    finally:
        pyboy.stop(save=False)
    if derive_moves:
        env = PokemonBattleEnv(rom, output)
        try:
            env.reset(options={'scenario': 0})
            for i in range(derive_moves):
                _, _, terminated, truncated, info = env.step(0)
                if terminated or truncated:
                    break
                buffer = io.BytesIO()
                env.pyboy.save_state(buffer)
                snapshot = buffer.getvalue()
                filename = f'battle_{i + 2:04d}.state'
                (output / filename).write_bytes(snapshot)
                manifest['scenarios'].append({'snapshot': filename, 'snapshot_sha256': digest(snapshot),
                    **describe(env.pyboy.memory, lookups), 'provenance': {'kind': 'derived_from_battle_0001',
                    'move_1_actions': i + 1, 'additional_emulator_frames': env.controller.frames,
                    'additional_controller_inputs': env.controller.inputs}})
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        finally:
            env.close()
    print(f"Captured {len(manifest['scenarios'])} battle decision state(s) in {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rom', type=Path, default=ROOT / 'PokemonRed.gb')
    parser.add_argument('--start-state', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--policy', default='ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wait-frames', type=int, default=0)
    parser.add_argument('--wait-before-action', type=int, default=1)
    parser.add_argument('--derive-moves', type=int, default=0)
    args = parser.parse_args()
    capture(args.rom, args.start_state, args.trace, args.output, args.policy, args.seed,
            args.wait_frames, args.derive_moves, args.wait_before_action)


if __name__ == '__main__':
    main()
