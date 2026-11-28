"""Train a standalone PPO baseline or the three-start adaptive curriculum."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from pokemon_game.adaptive_curriculum import AdaptiveStarts, SnapshotArchive, role_for_worker, state_info

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(path, record):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


def make_env(config, archive_dir, role):
    def create():
        from red_gym_env_v2 import RedGymEnv
        env = RedGymEnv(config)
        env.pyboy.set_emulation_speed(0)
        return AdaptiveStarts(env, config['init_state'], archive_dir, role)
    return create


def evaluate(model, env, max_steps):
    started = time.perf_counter()
    obs, reset_info = env.reset()
    reward_total = 0.0
    for steps in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        reward_total += float(reward)
        if terminated or truncated:
            break
    return {'steps': steps, 'original_return': reward_total,
            'start': reset_info['curriculum_start'], 'final': state_info(env.unwrapped.pyboy.memory),
            'badges': int(env.unwrapped.get_badges()),
            'evaluation_wall_seconds': time.perf_counter() - started}


class Recorder(BaseCallback):
    def __init__(self, output, archive, condition, eval_envs, max_steps, eval_every, started):
        super().__init__()
        self.output, self.archive, self.condition = output, archive, condition
        self.eval_envs, self.max_steps, self.eval_every = eval_envs, max_steps, eval_every
        self.started, self.rollouts, self.candidates = started, 0, []
        self.completed = 0

    def record_eval(self):
        for label, env in self.eval_envs:
            result = evaluate(self.model, env, self.max_steps)
            append(self.output / 'evaluation.jsonl', {'timesteps': self.num_timesteps, 'save': label,
                'elapsed_wall_seconds': time.perf_counter() - self.started, **result})

    def _on_training_start(self):
        self.record_eval()

    def _on_rollout_start(self):
        if self.rollouts and self.rollouts % self.eval_every == 0:
            self.record_eval()

    def _on_step(self):
        for info in self.locals['infos']:
            candidate = info.get('curriculum_episode')
            if candidate is not None:
                candidate['timesteps'] = self.num_timesteps
                self.candidates.append(candidate)
                self.completed += 1
                append(self.output / 'episodes.jsonl',
                       {key: value for key, value in candidate.items() if key != 'snapshot'})
        return True

    def _on_rollout_end(self):
        self.rollouts += 1
        published = (self.archive.publish(self.candidates, self.rollouts)
                     if self.condition == 'adaptive' else None)
        append(self.output / 'training.jsonl', {'rollout': self.rollouts, 'timesteps': self.num_timesteps,
            'elapsed_wall_seconds': time.perf_counter() - self.started, 'completed_episodes': self.completed,
            'rollout_episodes': len(self.candidates), 'published': published})
        self.candidates = []
        print(f'Rollout {self.rollouts}: {self.num_timesteps} transitions', flush=True)

    def _on_training_end(self):
        self.record_eval()


def run(args):
    if args.workers < 4 or args.workers % 4 or args.n_steps < 2 or args.rollouts < 1:
        raise ValueError('Workers must be a multiple of four; rollout limits must be positive')
    if args.batch_size < 2 or args.workers * args.n_steps % args.batch_size:
        raise ValueError('Batch size must divide the rollout size')
    if args.max_steps < 1 or args.eval_every < 1:
        raise ValueError('Invalid episode or evaluation interval')
    output, start, rom = args.output.resolve(), args.start_state.resolve(), args.rom.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    eval_states = [('initial', start)] + [(f'heldout_{i}', p.resolve()) for i, p in enumerate(args.eval_state, 1)]
    if any(digest(path) == digest(start) for _, path in eval_states[1:]):
        raise ValueError('Held-out save must differ from the initial save')
    output.mkdir(parents=True)
    archive = SnapshotArchive(output / 'archive')
    started = time.perf_counter()
    torch.set_num_threads(1)
    config = {'headless': True, 'save_final_state': False, 'early_stop': False, 'action_freq': 24,
        'init_state': str(start), 'max_steps': args.max_steps, 'print_rewards': False, 'save_video': False,
        'fast_video': True, 'session_path': output / 'sessions', 'gb_path': str(rom), 'debug': False,
        'reward_scale': 0.5, 'explore_weight': 0.25,
        'rewards': {'level': True, 'opponent_level': True, 'death_penalty': True}}
    config['session_path'].mkdir()
    roles = [role_for_worker(i, args.workers) if args.condition == 'adaptive' else 'initial'
             for i in range(args.workers)]
    protocol = {'condition': args.condition, 'seed': args.seed, 'workers': args.workers,
        'roles': roles, 'rollout_steps': args.n_steps, 'rollouts': args.rollouts,
        'total_timesteps': args.workers * args.n_steps * args.rollouts, 'batch_size': args.batch_size,
        'epochs': 1, 'gamma': 0.997, 'learning_rate': 0.0003, 'ent_coef': 0.01,
        'episode_limit': args.max_steps, 'policy': 'MultiInputPolicy from scratch',
        'base_rewards': config['rewards'], 'rom_sha256': digest(rom), 'initial_sha256': digest(start),
        'eval_sha256': {label: digest(path) for label, path in eval_states},
        'selection': 'highest and lowest original episode return among eligible completions in the last rollout',
        'initial_fallback': 'before first publication and on invalid/missing/corrupt snapshots',
        'eligibility': 'finite return, snapshot bytes, valid party count and battle flag; not proof of recoverability',
        'snapshot_updates': 'published at rollout boundary; adopted by each worker at its next reset',
        'evaluation': 'one deterministic episode per fixed save; not an independent success-rate estimate',
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
            (ROOT / 'pokemon_game/adaptive_curriculum.py', ROOT / 'pokemon_game/curriculum_experiment.py',
             ROOT / 'v2/red_gym_env_v2.py')}}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    sys.path.insert(0, str(ROOT / 'v2'))
    os.chdir(ROOT / 'v2')
    env = None
    eval_envs = []
    try:
        env = SubprocVecEnv([make_env(config, archive.directory, role) for role in roles], start_method='forkserver')
        for label, path in eval_states:
            eval_config = {**config, 'init_state': str(path)}
            eval_envs.append((label, make_env(eval_config, archive.directory, 'initial')()))
        model = PPO('MultiInputPolicy', env, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=1,
                    gamma=0.997, learning_rate=0.0003, ent_coef=0.01, seed=args.seed, device='cpu', verbose=0)
        recorder = Recorder(output, archive, args.condition, eval_envs, args.max_steps, args.eval_every, started)
        model.learn(total_timesteps=protocol['total_timesteps'], callback=recorder)
        model.save(str(output / 'final_model'))
        (output / 'complete.json').write_text(json.dumps({'timesteps': model.num_timesteps,
            'wall_seconds': time.perf_counter() - started}, indent=2) + '\n')
    finally:
        for _, eval_env in eval_envs:
            eval_env.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--condition', choices=('control', 'adaptive'), required=True)
    parser.add_argument('--start-state', type=Path, default=ROOT / 'init.state')
    parser.add_argument('--rom', type=Path, default=ROOT / 'PokemonRed.gb')
    parser.add_argument('--eval-state', type=Path, action='append', default=[])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--n-steps', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--rollouts', type=int, default=61)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--eval-every', type=int, default=5)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
