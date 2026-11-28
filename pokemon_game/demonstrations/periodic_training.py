"""Run optional cloning before and between completed PPO update cycles."""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pokemon_game.demonstrations.behavioral_cloning import accuracy, clone, validate_split
from pokemon_game.demonstrations.dataset import DemonstrationDataset
from pokemon_game.demonstrations.environment import ROOT, enter_v2, make_env


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def run_schedule(model, dataset, validation, config, rollouts, output, seed=0, dataset_loader=None):
    """Each PPO chunk ends with train(); cloning precedes a fresh next rollout."""
    imitation = config['imitation_learning']
    initial = imitation['initial_pretraining']
    periodic = imitation['periodic_training']
    enabled = bool(imitation['enabled'])
    interval = int(periodic['interval_rollouts'])
    if rollouts < 1 or interval < 1:
        raise ValueError('Rollouts and imitation interval must be positive')
    if enabled and dataset is None:
        raise ValueError('Imitation requires demonstrations')
    started = time.perf_counter()
    completed, cycles = 0, 0

    def update(kind, epochs):
        nonlocal cycles, dataset
        if dataset_loader is not None:
            dataset = dataset_loader()
        before = accuracy(model, dataset, int(imitation['batch_size']))
        losses = clone(model, dataset, epochs=int(epochs), batch_size=int(imitation['batch_size']),
                       learning_rate=float(imitation['learning_rate']), seed=seed + cycles)
        cycles += 1
        event = {'phase': kind, 'completed_rollouts': completed, 'timesteps': model.num_timesteps,
                 'training_accuracy_before': before,
                 'training_accuracy_after': accuracy(model, dataset, int(imitation['batch_size'])),
                 'validation_accuracy': accuracy(model, validation, int(imitation['batch_size'])) if validation else None,
                 'demonstrations': dataset.sessions, 'demonstration_transitions': len(dataset),
                 'losses': losses, 'wall_seconds': time.perf_counter() - started}
        append(output / 'schedule.jsonl', event)
        model.save(str(output / f'imitation-{cycles:03d}'))
        return event

    if enabled and initial['enabled']:
        update('initial_pretraining', initial['epochs'])
    while completed < rollouts:
        chunk = min(interval if enabled and periodic['enabled'] else rollouts, rollouts - completed)
        # Stable-Baselines3 completes PPO.train() before learn() returns.
        old_steps = model.num_timesteps
        model.learn(total_timesteps=chunk * model.n_steps * model.n_envs, reset_num_timesteps=False)
        expected = old_steps + chunk * model.n_steps * model.n_envs
        if model.num_timesteps != expected:
            raise RuntimeError('PPO transition count did not match complete rollouts')
        completed += chunk
        append(output / 'schedule.jsonl', {'phase': 'ppo_update_complete', 'completed_rollouts': completed,
            'timesteps': model.num_timesteps, 'wall_seconds': time.perf_counter() - started})
        model.save(str(output / f'ppo-rollout-{completed:04d}'))
        if enabled and periodic['enabled'] and completed < rollouts and completed % interval == 0:
            update('periodic_imitation', periodic['epochs'])
    return {'rollouts': completed, 'training_timesteps': model.num_timesteps,
            'imitation_updates': cycles, 'wall_seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/imitation_example.yaml')
    parser.add_argument('--rom', type=Path, default=ROOT / 'PokemonRed.gb')
    parser.add_argument('--start-state', type=Path, default=ROOT / 'init.state')
    parser.add_argument('--goal-map-id', type=int)
    parser.add_argument('--sessions', nargs='*', type=Path, default=[])
    parser.add_argument('--demonstration-root', type=Path,
                        help='Reload finalized session directories from this folder at each imitation update')
    parser.add_argument('--validation-sessions', nargs='*', type=Path, default=[])
    parser.add_argument('--allow-mixed-goals', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output, rom, start = args.output.resolve(), args.rom.resolve(), args.start_state.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    explicit_sessions = [path.resolve() for path in args.sessions]
    validation_sessions = [path.resolve() for path in args.validation_sessions]
    demo_root = args.demonstration_root.resolve(strict=True) if args.demonstration_root else None
    config = yaml.safe_load(args.config.read_text())
    training, imitation = config['training'], config['imitation_learning']
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    if int(training['workers']) < 1 or int(training['rollout_steps']) < 2:
        raise ValueError('Invalid number of workers or rollout steps')
    if args.goal_map_id is not None and not 0 <= args.goal_map_id <= 255:
        raise ValueError('Goal map ID must be a byte')
    sources = [name for name, selected in imitation['sources'].items() if selected]
    validation = (DemonstrationDataset(validation_sessions, sources,
                                       allow_mixed_goals=args.allow_mixed_goals)
                  if imitation['enabled'] and validation_sessions else None)

    def load_demonstrations():
        paths = list(explicit_sessions)
        if demo_root is not None:
            for manifest_path in sorted(demo_root.glob('*/manifest.json')):
                manifest = json.loads(manifest_path.read_text())
                source, outcome = manifest.get('source'), manifest.get('outcome')
                finished = outcome not in ('recording', 'interrupted')
                successful = source not in ('greedy_expert', 'successful_ppo') or outcome in ('success', 'goal_reached')
                path = manifest_path.parent.resolve()
                if finished and successful and source in sources and path not in paths:
                    paths.append(path)
        result = DemonstrationDataset(paths, sources, allow_mixed_goals=args.allow_mixed_goals)
        validate_split(result, validation)
        if args.goal_map_id is not None and result.goals != {args.goal_map_id} and not args.allow_mixed_goals:
            raise ValueError('Demonstration goal does not match the training goal')
        return result

    dataset = load_demonstrations() if imitation['enabled'] else None
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    protocol = {'config': config, 'goal_map_id': args.goal_map_id, 'demonstrations': dataset.sessions if dataset else [],
                'validation': validation.sessions if validation else [], 'allow_mixed_goals': args.allow_mixed_goals,
                'demonstration_root': str(demo_root) if demo_root else None,
                'checkpoint': str(checkpoint) if checkpoint else None,
                'schedule': 'cloning after completed PPO learn/update cycles; fresh rollouts follow',
                'bc_optimizer': 'separate Adam, reinitialized per cloning update',
                'ppo_learning_rate_schedule': 'constant; each PPO chunk calls learn separately'}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    enter_v2()
    factories = [lambda rank=rank: make_env(rom, start, output / 'sessions' / str(rank),
                   int(training['max_episode_steps']), args.goal_map_id)
                 for rank in range(int(training['workers']))]
    env = None
    try:
        env = DummyVecEnv(factories) if len(factories) == 1 else SubprocVecEnv(factories, start_method='forkserver')
        if checkpoint:
            model = PPO.load(str(checkpoint), env=env, device='cpu')
            expected = {'n_steps': int(training['rollout_steps']), 'batch_size': int(training['batch_size']),
                        'n_epochs': int(training['ppo_epochs']), 'gamma': float(training['gamma']),
                        'learning_rate': float(training['learning_rate']), 'ent_coef': float(training['ent_coef'])}
            if any(getattr(model, key) != value for key, value in expected.items()):
                raise ValueError('Checkpoint PPO hyperparameters differ from the configuration')
        else:
            model = PPO('MultiInputPolicy', env, n_steps=int(training['rollout_steps']),
                batch_size=int(training['batch_size']), n_epochs=int(training['ppo_epochs']),
                gamma=float(training['gamma']), learning_rate=float(training['learning_rate']),
                ent_coef=float(training['ent_coef']), seed=int(training['seed']), device='cpu', verbose=0)
        result = run_schedule(model, dataset, validation, config, int(training['rollouts']), output,
                              seed=int(training['seed']), dataset_loader=load_demonstrations if demo_root else None)
        model.save(str(output / 'final_model'))
        (output / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
