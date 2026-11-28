"""Run optional cloning before and between completed PPO update cycles."""
import argparse
import hashlib
import json
import time
from collections import deque
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pokemon_game.demonstrations.behavioral_cloning import accuracy, clone, validate_split
from pokemon_game.demonstrations.dataset import DemonstrationDataset
from pokemon_game.demonstrations.environment import ROOT, enter_v2, make_env


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


class TrainingProgress(BaseCallback):
    """Report collected rollouts and completed-episode statistics as they happen."""
    def __init__(self, output, total_rollouts, started, initial_steps, goal_enabled):
        super().__init__(verbose=0)
        self.output, self.total_rollouts = output, total_rollouts
        self.started, self.initial_steps, self.goal_enabled = started, initial_steps, goal_enabled
        self.episode_returns = None
        self.episode_lengths = None
        self.recent_episodes = deque(maxlen=100)
        self.episodes = 0

    def _on_step(self):
        rewards, dones, infos = self.locals['rewards'], self.locals['dones'], self.locals['infos']
        if self.episode_returns is None:
            self.episode_returns = [0.0] * len(rewards)
            self.episode_lengths = [0] * len(rewards)
        for index, (reward, done, info) in enumerate(zip(rewards, dones, infos)):
            self.episode_returns[index] += float(info.get('original_reward', reward))
            self.episode_lengths[index] += 1
            if done:
                self.recent_episodes.append({'return': self.episode_returns[index],
                    'length': self.episode_lengths[index], 'goal_reached': bool(info.get('goal_reached', False))})
                self.episodes += 1
                self.episode_returns[index] = 0.0
                self.episode_lengths[index] = 0
        return True

    def _on_rollout_end(self):
        steps_per_rollout = self.model.n_steps * self.model.n_envs
        collected = self.num_timesteps - self.initial_steps
        rollout = collected // steps_per_rollout
        count = len(self.recent_episodes)
        mean_return = sum(e['return'] for e in self.recent_episodes) / count if count else None
        mean_length = sum(e['length'] for e in self.recent_episodes) / count if count else None
        successes = sum(e['goal_reached'] for e in self.recent_episodes) if self.goal_enabled else None
        event = {'phase': 'rollout_collected', 'rollout': rollout, 'total_rollouts': self.total_rollouts,
                 'timesteps': self.num_timesteps, 'transitions_this_run': collected,
                 'total_transitions_this_run': self.total_rollouts * steps_per_rollout,
                 'completed_episodes': self.episodes, 'recent_episode_count': count,
                 'mean_original_return': mean_return, 'mean_episode_length': mean_length,
                 'goal_successes': successes, 'wall_seconds': time.perf_counter() - self.started}
        append(self.output / 'progress.jsonl', event)
        details = (f"last {count} episodes: mean return {mean_return:.3f}, "
                   f"mean length {mean_length:.1f}" if count else 'no completed episodes yet')
        if successes is not None and count:
            details += f', goal successes {successes}/{count}'
        print(f"[PPO] Rollout {rollout}/{self.total_rollouts} collected; "
              f"{collected:,}/{self.total_rollouts * steps_per_rollout:,} transitions; "
              f"{details}; {event['wall_seconds']:.1f}s elapsed. PPO update next.", flush=True)


def run_schedule(model, dataset, validation, config, rollouts, output, seed=0, dataset_loader=None,
                 show_progress=False, goal_enabled=False, evaluator=None):
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
    progress = (TrainingProgress(output, rollouts, started, model.num_timesteps, goal_enabled)
                if show_progress else None)

    def update(kind, epochs):
        nonlocal cycles, dataset
        if dataset_loader is not None:
            dataset = dataset_loader()
        print(f"[Imitation] Starting {kind.replace('_', ' ')}: {epochs} epochs on "
              f"{len(dataset):,} transitions at PPO step {model.num_timesteps:,}.", flush=True)
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
        validation_text = (f", validation agreement {event['validation_accuracy']:.1%}"
                           if event['validation_accuracy'] is not None else '')
        print(f"[Imitation] Complete: action agreement {before:.1%} → "
              f"{event['training_accuracy_after']:.1%}{validation_text}; "
              f"loss {losses[-1]:.4f}; saved imitation-{cycles:03d}.zip.", flush=True)
        return event

    if enabled and initial['enabled']:
        update('initial_pretraining', initial['epochs'])
    if evaluator is not None:
        evaluator(model, completed, 'before_ppo')
    while completed < rollouts:
        # Keep evaluation checkpoints matched even for the no-imitation control.
        chunk = min(interval if (enabled and periodic['enabled']) or evaluator is not None else rollouts,
                    rollouts - completed)
        # Stable-Baselines3 completes PPO.train() before learn() returns.
        old_steps = model.num_timesteps
        learn_kwargs = {'total_timesteps': chunk * model.n_steps * model.n_envs,
                        'reset_num_timesteps': False}
        if progress is not None:
            learn_kwargs['callback'] = progress
        model.learn(**learn_kwargs)
        expected = old_steps + chunk * model.n_steps * model.n_envs
        if model.num_timesteps != expected:
            raise RuntimeError('PPO transition count did not match complete rollouts')
        completed += chunk
        append(output / 'schedule.jsonl', {'phase': 'ppo_update_complete', 'completed_rollouts': completed,
            'timesteps': model.num_timesteps, 'wall_seconds': time.perf_counter() - started})
        model.save(str(output / f'ppo-rollout-{completed:04d}'))
        print(f"[PPO] Updates through rollout {completed}/{rollouts} complete; "
              f"saved ppo-rollout-{completed:04d}.zip.", flush=True)
        if enabled and periodic['enabled'] and completed < rollouts and completed % interval == 0:
            update('periodic_imitation', periodic['epochs'])
        if evaluator is not None:
            evaluator(model, completed, 'after_update')
    return {'rollouts': completed, 'training_timesteps': model.num_timesteps,
            'imitation_updates': cycles, 'wall_seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/imitation_example.yaml')
    parser.add_argument('--rom', type=Path, default=ROOT / 'PokemonRed.gb')
    parser.add_argument('--start-state', type=Path, default=ROOT / 'init.state')
    parser.add_argument('--goal-map-id', type=int)
    parser.add_argument('--goal-map-ids', nargs='+', type=int,
                        help='Condition observations on a sampled map goal from this list')
    parser.add_argument('--sessions', nargs='*', type=Path, default=[])
    parser.add_argument('--demonstration-root', type=Path,
                        help='Reload finalized session directories from this folder at each imitation update')
    parser.add_argument('--validation-sessions', nargs='*', type=Path, default=[])
    parser.add_argument('--evaluation-states', nargs='*', type=Path, default=[],
                        help='Fixed emulator saves for deterministic goal evaluations at PPO chunk boundaries')
    parser.add_argument('--evaluation-goal-map-id', type=int,
                        help='Goal map for evaluations; defaults to the single training goal')
    parser.add_argument('--allow-mixed-goals', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output, rom, start = args.output.resolve(), args.rom.resolve(), args.start_state.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    explicit_sessions = [path.resolve() for path in args.sessions]
    validation_sessions = [path.resolve() for path in args.validation_sessions]
    evaluation_states = [path.resolve() for path in args.evaluation_states]
    demo_root = args.demonstration_root.resolve(strict=True) if args.demonstration_root else None
    config = yaml.safe_load(args.config.read_text())
    training, imitation = config['training'], config['imitation_learning']
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    if int(training['workers']) < 1 or int(training['rollout_steps']) < 2:
        raise ValueError('Invalid number of workers or rollout steps')
    if args.goal_map_id is not None and not 0 <= args.goal_map_id <= 255:
        raise ValueError('Goal map ID must be a byte')
    if args.goal_map_ids is not None:
        if args.goal_map_id is not None:
            raise ValueError('Use either --goal-map-id or --goal-map-ids')
        if len(set(args.goal_map_ids)) != len(args.goal_map_ids) or any(not 0 <= value <= 255 for value in args.goal_map_ids):
            raise ValueError('Goal map IDs must be distinct bytes')
    goal_conditioned = args.goal_map_ids is not None
    evaluation_goal = args.evaluation_goal_map_id
    if evaluation_goal is None:
        evaluation_goal = args.goal_map_id if not goal_conditioned else (args.goal_map_ids[0] if len(args.goal_map_ids) == 1 else None)
    if evaluation_states and (evaluation_goal is None or not 0 <= evaluation_goal <= 255):
        raise ValueError('Evaluation states require a byte-valued evaluation goal')
    if goal_conditioned and evaluation_states and evaluation_goal not in args.goal_map_ids:
        raise ValueError('Evaluation goal must be one of the configured goals')
    sources = [name for name, selected in imitation['sources'].items() if selected]
    validation = (DemonstrationDataset(validation_sessions, sources,
                                       allow_mixed_goals=args.allow_mixed_goals,
                                       goal_conditioned=goal_conditioned)
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
        result = DemonstrationDataset(paths, sources, allow_mixed_goals=args.allow_mixed_goals,
                                      goal_conditioned=goal_conditioned)
        validate_split(result, validation)
        if args.goal_map_id is not None and result.goals != {args.goal_map_id} and not args.allow_mixed_goals:
            raise ValueError('Demonstration goal does not match the training goal')
        if goal_conditioned and not result.goals <= set(args.goal_map_ids):
            raise ValueError('Demonstration contains a goal absent from the training goals')
        return result

    print('[Setup] Loading demonstrations and initializing environments...', flush=True)
    dataset = load_demonstrations() if imitation['enabled'] else None
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    protocol = {'config': config, 'goal_map_id': args.goal_map_id, 'goal_map_ids': args.goal_map_ids,
                'goal_conditioned': goal_conditioned, 'demonstrations': dataset.sessions if dataset else [],
                'validation': validation.sessions if validation else [], 'allow_mixed_goals': args.allow_mixed_goals,
                'demonstration_root': str(demo_root) if demo_root else None,
                'evaluation_goal_map_id': evaluation_goal,
                'evaluation_state_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                            for path in evaluation_states},
                'checkpoint': str(checkpoint) if checkpoint else None,
                'schedule': 'cloning after completed PPO learn/update cycles; fresh rollouts follow',
                'bc_optimizer': 'separate Adam, reinitialized per cloning update',
                'ppo_learning_rate_schedule': 'constant; each PPO chunk calls learn separately'}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    enter_v2()
    factories = [lambda rank=rank: make_env(rom, start, output / 'sessions' / str(rank),
                   int(training['max_episode_steps']), args.goal_map_id, args.goal_map_ids)
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

        def evaluate_fixed(policy, completed, phase):
            for index, path in enumerate(evaluation_states):
                evaluation_env = make_env(rom, path, output / 'evaluation-sessions' / str(index),
                    int(training['max_episode_steps']),
                    target_map=None if goal_conditioned else evaluation_goal,
                    goal_maps=[evaluation_goal] if goal_conditioned else None)
                try:
                    try:
                        obs, _ = evaluation_env.reset()
                    except ValueError:
                        if int(evaluation_env.unwrapped.pyboy.memory[0xD35E]) != evaluation_goal:
                            raise
                        obs = None
                    start_map = int(evaluation_env.unwrapped.pyboy.memory[0xD35E])
                    total_reward, steps, success, outcome = 0.0, 0, False, 'step_limit'
                    if start_map == evaluation_goal:
                        outcome = 'already_at_goal'
                    else:
                        for _ in range(int(training['max_episode_steps'])):
                            action, _ = policy.predict(obs, deterministic=True)
                            obs, reward, terminated, truncated, info = evaluation_env.step(int(action))
                            steps += 1
                            total_reward += float(reward)
                            if info.get('goal_reached'):
                                success, outcome = True, 'goal_reached'
                                break
                            if terminated or truncated:
                                outcome = 'environment_ended'
                                break
                    row = {'phase': phase, 'completed_rollouts': completed, 'timesteps': policy.num_timesteps,
                           'save_index': index, 'start_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'goal_map_id': evaluation_goal, 'outcome': outcome, 'goal_reached': success,
                           'steps': steps, 'original_return': total_reward,
                           'final_map_id': int(evaluation_env.unwrapped.pyboy.memory[0xD35E])}
                    append(output / 'evaluation.jsonl', row)
                    print(f"[Evaluation] Save {index}, PPO step {policy.num_timesteps:,}: "
                          f"{outcome} after {steps} actions.", flush=True)
                finally:
                    evaluation_env.close()

        result = run_schedule(model, dataset, validation, config, int(training['rollouts']), output,
                              seed=int(training['seed']), dataset_loader=load_demonstrations if demo_root else None,
                              show_progress=True, goal_enabled=(args.goal_map_id is not None or goal_conditioned),
                              evaluator=evaluate_fixed if evaluation_states else None)
        model.save(str(output / 'final_model'))
        (output / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
