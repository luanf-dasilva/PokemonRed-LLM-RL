"""Sequential matched-seed curriculum and imitation PPO comparisons."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def write_status(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--start-state', required=True, type=Path)
    parser.add_argument('--eval-state', required=True, type=Path)
    parser.add_argument('--expert-session', required=True, type=Path)
    parser.add_argument('--human-session', required=True, type=Path)
    parser.add_argument('--validation-session', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--n-steps', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--rollouts', type=int, default=61)
    parser.add_argument('--max-steps', type=int, default=300)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.workers < 4 or args.workers % 4:
        raise ValueError('Seeds must be unique and workers a positive multiple of four')
    if args.n_steps < 2 or args.batch_size < 2 or args.n_steps * args.workers % args.batch_size or args.rollouts < 1:
        raise ValueError('Invalid rollout, batch size or budget')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    paths = {name: getattr(args, name).resolve(strict=True) for name in
             ('start_state', 'eval_state', 'expert_session', 'human_session', 'validation_session')}
    output.mkdir(parents=True)
    plan = []
    for seed in args.seeds:
        for condition in ('imitation-control', 'imitation-assisted', 'curriculum-control', 'curriculum-adaptive'):
            label = f'{condition}-seed-{seed}'
            if condition.startswith('imitation-'):
                config = yaml.safe_load((ROOT / 'configs/imitation_example.yaml').read_text())
                config['training'].update({'workers': args.workers, 'rollout_steps': args.n_steps,
                    'batch_size': args.batch_size, 'rollouts': args.rollouts,
                    'max_episode_steps': args.max_steps, 'seed': seed})
                config['imitation_learning']['enabled'] = condition == 'imitation-assisted'
                config_path = output / (label + '.yaml')
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                command = [sys.executable, '-m', 'pokemon_game.demonstrations.periodic_training',
                    '--config', str(config_path), '--output', str(output / label),
                    '--start-state', str(paths['start_state']), '--goal-map-id', '1',
                    '--sessions', str(paths['expert_session']), str(paths['human_session']),
                    '--validation-sessions', str(paths['validation_session']),
                    '--evaluation-states', str(paths['start_state']), str(paths['eval_state'])]
            else:
                command = [sys.executable, '-m', 'pokemon_game.curriculum_experiment',
                    '--condition', condition.removeprefix('curriculum-'), '--seed', str(seed),
                    '--output', str(output / label), '--start-state', str(paths['start_state']),
                    '--goal-map-id', '1', '--eval-state', str(paths['eval_state']),
                    '--workers', str(args.workers), '--n-steps', str(args.n_steps),
                    '--batch-size', str(args.batch_size), '--rollouts', str(args.rollouts),
                    '--max-steps', str(args.max_steps)]
            plan.append({'label': label, 'condition': condition, 'seed': seed, 'command': command})
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    completed = []
    status = output / 'status.json'
    for entry in plan:
        write_status(status, {'status': 'running', 'current': entry['label'], 'completed': completed, 'total': len(plan)})
        print(f"Starting {entry['label']} ({len(completed) + 1}/{len(plan)})", flush=True)
        with (output / (entry['label'] + '.log')).open('w') as stream:
            result = subprocess.run(entry['command'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            write_status(status, {'status': 'failed', 'current': entry['label'], 'completed': completed,
                                  'total': len(plan), 'returncode': result.returncode})
            raise SystemExit(f"Failed: {entry['label']} (exit {result.returncode})")
        completed.append(entry['label'])
    write_status(status, {'status': 'complete', 'current': None, 'completed': completed, 'total': len(plan)})
    print('Independent comparisons completed', flush=True)


if __name__ == '__main__':
    main()
