"""Plot per-seed evaluation outcomes and summarize completed PPO runs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matrix', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--save', default='training_save', help='Evaluation save label from protocol')
    parser.add_argument('--include-incomplete', action='store_true')
    parser.add_argument('--include-graph-cost', action='store_true',
                        help='Add recorded graph discovery seconds to tile wall time')
    args = parser.parse_args()
    plan = json.loads((args.matrix / 'plan.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate, runs = defaultdict(list), []
    for item in plan:
        path = args.matrix / item['label']
        if not (path / 'protocol.json').exists() or (not args.include_incomplete and not (path / 'complete.json').exists()):
            continue
        protocol = json.loads((path / 'protocol.json').read_text())
        evals = [row for row in read_lines(path / 'evaluation.jsonl') if row['save'] == args.save]
        if not evals:
            continue
        condition = f"{item['condition']} β={item['beta']:g}"
        for row in evals:
            aggregate[(condition, row['timesteps'])].append(int(row['success']))
        episodes = read_lines(path / 'episodes.jsonl')
        successes = [row['timesteps'] for row in episodes if row['success']]
        sustained = next((evals[i]['timesteps'] for i in range(len(evals) - 2)
                          if all(row['success'] for row in evals[i:i + 3])), None)
        runs.append((condition, item['seed'], evals, min(successes) if successes else None,
                     sustained, protocol.get('graph_construction_seconds') or 0.0))
    if not runs:
        raise ValueError('No matching runs with evaluation data')
    with (args.output / 'evaluation-summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['condition', 'timesteps', 'seeds', 'successful_seeds', 'success_fraction'])
        for (condition, step), outcomes in sorted(aggregate.items()):
            writer.writerow([condition, step, len(outcomes), sum(outcomes), sum(outcomes) / len(outcomes)])
    with (args.output / 'seed-summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['condition', 'seed', 'first_training_success', 'sustained_eval_success'])
        for condition, seed, _, first, sustained, _ in runs:
            writer.writerow([condition, seed, first, sustained])
    fig, ax = plt.subplots(figsize=(9, 5))
    for condition in sorted({key[0] for key in aggregate}):
        points = sorted((step, sum(values) / len(values)) for (name, step), values in aggregate.items()
                        if name == condition)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker='.', label=condition)
    ax.set(xlabel='Training transitions', ylabel='Fraction of seeds reaching goal', ylim=(-0.05, 1.05),
           title=f'Deterministic evaluation: {args.save}')
    ax.legend(fontsize='small')
    fig.tight_layout()
    fig.savefig(args.output / 'success-vs-transitions.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for condition, seed, evals, _, _, graph_seconds in runs:
        offset = graph_seconds if args.include_graph_cost else 0.0
        ax.step([row['elapsed_wall_seconds'] + offset for row in evals],
                [int(row['success']) for row in evals], where='post', label=f'{condition}, seed {seed}')
    ax.set(xlabel='Elapsed wall seconds' + (' (tile graph cost included)' if args.include_graph_cost else ''),
           ylabel='Goal reached (one episode)', ylim=(-0.05, 1.05),
           title=f'Per-seed deterministic evaluation: {args.save}')
    ax.legend(fontsize='x-small', ncol=2)
    fig.tight_layout()
    fig.savefig(args.output / 'success-vs-wall-time.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
