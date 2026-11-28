"""Evaluate simple semantic policies on a fixed battle archive; no training."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from pokemon_game.battle_env import PokemonBattleEnv


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(rom, archive, output, policy, repetitions, seed, max_actions):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    if repetitions < 1:
        raise ValueError('Repetitions must be positive')
    env = PokemonBattleEnv(rom, archive, max_actions=max_actions)
    rows = []
    try:
        output.mkdir(parents=True)
        protocol = {'policy': policy, 'repetitions_per_scenario': repetitions, 'seed': seed,
            'max_abstract_actions': max_actions, 'rom_sha256': digest(rom),
            'archive_manifest_sha256': digest(Path(archive) / 'manifest.json'),
            'limitations': ['Snapshots may share a parent encounter or exploration save',
                            'Repetitions reload identical emulator RNG states',
                            'Policies are hand-written, not trained PPO agents']}
        (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        rng = random.Random(seed)
        for scenario in range(len(env.scenarios)):
            for repetition in range(repetitions):
                _, info = env.reset(options={'scenario': scenario})
                actions, total_reward = [], 0.0
                start = time.perf_counter()
                while True:
                    valid = [i for i, enabled in enumerate(info['action_mask']) if enabled]
                    action = (next((i for i in valid if i < 4), 4) if policy == 'first_move'
                              else rng.choice(valid))
                    _, reward, terminated, truncated, info = env.step(action)
                    actions.append(action)
                    total_reward += reward
                    if terminated or truncated:
                        break
                row = {'scenario': scenario, 'repetition': repetition, 'outcome': info['outcome'],
                    'terminated': terminated, 'truncated': truncated, 'abstract_actions': len(actions),
                    'actions': actions, 'emulator_frames': env.controller.frames,
                    'controller_inputs': env.controller.inputs, 'reward': total_reward,
                    'wall_seconds': time.perf_counter() - start}
                rows.append(row)
                with (output / 'results.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
        counts = {}
        for row in rows:
            counts[row['outcome']] = counts.get(row['outcome'], 0) + 1
        summary = {'episodes': len(rows), 'outcomes': counts,
                   'mean_abstract_actions': sum(row['abstract_actions'] for row in rows) / len(rows),
                   'mean_emulator_frames': sum(row['emulator_frames'] for row in rows) / len(rows),
                   'mean_controller_inputs': sum(row['controller_inputs'] for row in rows) / len(rows)}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rom', type=Path, default=Path(__file__).resolve().parent.parent / 'PokemonRed.gb')
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--policy', choices=('first_move', 'random_legal'), default='first_move')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-actions', type=int, default=30)
    args = parser.parse_args()
    run(args.rom, args.archive, args.output, args.policy, args.repetitions, args.seed, args.max_actions)


if __name__ == '__main__':
    main()
