"""Run the preregistered six-condition, three-seed navigation PPO matrix."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = [('modified', 0.0), ('tile', 1.0), ('upstream', 0.0),
              ('map', 1.0), ('tile', 0.1), ('tile', 5.0)]


def write_status(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-state', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--eval-state', type=Path, action='append', default=[])
    parser.add_argument('--graph-discovery-seconds', type=float, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start, graph = args.start_state.resolve(strict=True), args.graph.resolve(strict=True)
    eval_states = [p.resolve(strict=True) for p in args.eval_state]
    if args.graph_discovery_seconds < 0:
        raise ValueError('Discovery time must be nonnegative')
    plan = []
    for seed in (0, 1, 2):
        for condition, beta in CONDITIONS:
            label = f'{condition}-beta-{beta:g}-seed-{seed}'
            command = [sys.executable, '-m', 'pokemon_game.ppo_navigation_experiment',
                '--output', str(output / label), '--condition', condition, '--beta', str(beta),
                '--seed', str(seed), '--start-state', str(start), '--graph', str(graph),
                '--graph-discovery-seconds', str(args.graph_discovery_seconds)]
            for path in eval_states:
                command += ['--eval-state', str(path)]
            plan.append({'label': label, 'condition': condition, 'beta': beta,
                         'seed': seed, 'command': command})
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    status_path = output / 'matrix-status.json'
    completed = []
    for entry in plan:
        write_status(status_path, {'status': 'running', 'current': entry['label'],
                                   'completed': completed, 'total': len(plan)})
        print(f"Starting {entry['label']} ({len(completed) + 1}/{len(plan)})", flush=True)
        with (output / (entry['label'] + '.log')).open('w') as stream:
            process = subprocess.run(entry['command'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            write_status(status_path, {'status': 'failed', 'current': entry['label'],
                'returncode': process.returncode, 'completed': completed, 'total': len(plan)})
            raise SystemExit(f"Run failed: {entry['label']} (exit {process.returncode})")
        completed.append(entry['label'])
    write_status(status_path, {'status': 'complete', 'current': None,
                               'completed': completed, 'total': len(plan)})
    print('PPO matrix completed', flush=True)


if __name__ == '__main__':
    main()
