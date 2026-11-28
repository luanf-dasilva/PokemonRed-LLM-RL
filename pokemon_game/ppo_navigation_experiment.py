"""Matched fixed-goal PPO experiment with separately measured shaping rewards."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Avoid oversubscribing CPUs when many emulator workers run concurrently.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from pokemon_game.reward_shaping import MapPotential, PotentialShaping, TilePotential
from pokemon_game.state_encoder import MAP_ID, load_lookups
from pokemon_game.value_function import LocationValue

ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = {'upstream': (False, 'none'), 'modified': (True, 'none'),
              'map': (True, 'map'), 'tile': (True, 'tile')}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


class GoalEpisode(gym.Wrapper):
    """Terminate every condition on first entry to the fixed goal map."""
    def __init__(self, env, target_map):
        super().__init__(env)
        self.target_map = target_map

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        memory = self.unwrapped.pyboy.memory
        success = memory[MAP_ID] == self.target_map and not bool(memory[0xD057])
        info = dict(info)
        info['goal_reached'] = success
        if success:
            terminated, truncated = True, False
        return obs, reward, terminated, truncated, info

    def close(self):
        self.unwrapped.pyboy.stop(save=False)


class EpisodeAccounting(gym.Wrapper):
    def reset(self, **kwargs):
        self.original_return = self.shaping_return = 0.0
        self.length = self.missing = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.original_return += info['original_reward']
        self.shaping_return += info['navigation_shaping']
        self.missing += int(info['navigation_potential_missing'])
        self.length += 1
        if terminated or truncated:
            info['navigation_episode'] = {'success': info['goal_reached'], 'steps': self.length,
                'original_return': self.original_return, 'shaping_return': self.shaping_return,
                'missing_potential_steps': self.missing, 'terminated': terminated, 'truncated': truncated}
        return obs, reward, terminated, truncated, info


def make_env(config, potential, beta, gamma, target_map):
    def create():
        from red_gym_env_v2 import RedGymEnv
        env = RedGymEnv(config)
        env.pyboy.set_emulation_speed(0)
        return EpisodeAccounting(PotentialShaping(GoalEpisode(env, target_map), potential, beta, gamma))
    return create


def make_potential(kind, graph_path, rom, target_name):
    if kind == 'none':
        return MapPotential({i: 0.0 for i in range(256)})
    if kind == 'tile':
        return TilePotential(graph_path, rom)
    lookups = load_lookups(ROOT / 'pokemon_rag/data/pret-pokered/processed/lookups.json')
    edges = json.loads((ROOT / 'pokemon_rag/data/pret-pokered/processed/map_graph.json').read_text())['edges']
    value = LocationValue(lookups, edges)
    goal = {'goal_type': 'REACH_LOCATION', 'target': target_name}
    values = {}
    for name, record in lookups['maps'].items():
        try:
            values[record['id']] = value.evaluate({'map_name': name}, goal)
        except ValueError:
            pass
    return MapPotential(values)


def evaluate(model, env, eval_steps):
    started = time.perf_counter()
    obs, _ = env.reset()
    success = False
    for step in range(1, eval_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            success = info['goal_reached']
            break
    return {'success': bool(success), 'steps': step,
            'original_return': env.original_return, 'evaluation_wall_seconds': time.perf_counter() - started}


class Recorder(BaseCallback):
    def __init__(self, output, eval_envs, max_steps, eval_every, started):
        super().__init__(verbose=0)
        self.output, self.eval_envs, self.max_steps = output, eval_envs, max_steps
        self.eval_every, self.started = eval_every, started
        self.rollouts = 0
        self.original = self.shaping = 0.0
        self.missing = self.transitions = 0
        self.episode_count = self.successes = 0
        self.first_success = None

    def record_eval(self):
        for label, env in self.eval_envs:
            result = evaluate(self.model, env, self.max_steps)
            append_json(self.output / 'evaluation.jsonl', {'timesteps': self.num_timesteps,
                'elapsed_wall_seconds': time.perf_counter() - self.started, 'save': label, **result})

    def _on_training_start(self):
        self.record_eval()

    def _on_rollout_start(self):
        # This callback occurs after the previous PPO update.
        if self.rollouts and self.rollouts % self.eval_every == 0:
            self.record_eval()

    def _on_step(self):
        for info in self.locals['infos']:
            self.original += info['original_reward']
            self.shaping += info['navigation_shaping']
            self.missing += int(info['navigation_potential_missing'])
            self.transitions += 1
            episode = info.get('navigation_episode')
            if episode:
                self.episode_count += 1
                self.successes += int(episode['success'])
                if episode['success'] and self.first_success is None:
                    self.first_success = self.num_timesteps
                append_json(self.output / 'episodes.jsonl', {'timesteps': self.num_timesteps,
                    'wall_seconds': time.perf_counter() - self.started, **episode})
        return True

    def _on_rollout_end(self):
        self.rollouts += 1
        append_json(self.output / 'training.jsonl', {'rollout': self.rollouts, 'timesteps': self.num_timesteps,
            'wall_seconds': time.perf_counter() - self.started, 'episodes': self.episode_count,
            'successes': self.successes, 'first_success_timesteps': self.first_success,
            'original_reward_sum': self.original, 'shaping_reward_sum': self.shaping,
            'missing_potential_steps': self.missing, 'transitions': self.transitions})
        print(f"Rollout {self.rollouts}: {self.num_timesteps} transitions, "
              f"{time.perf_counter() - self.started:.1f}s elapsed", flush=True)

    def _on_training_end(self):
        # Captures the final update, even when the budget is an eval interval.
        self.record_eval()


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    if args.workers < 1 or args.n_steps < 2 or args.rollouts < 1 or args.eval_every < 1 or args.max_steps < 1:
        raise ValueError('Invalid experiment limits')
    if args.batch_size < 2 or args.workers * args.n_steps % args.batch_size:
        raise ValueError('Batch size must divide the rollout size')
    modified, kind = CONDITIONS[args.condition]
    beta = args.beta if kind != 'none' else 0.0
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('Invalid beta')
    rom, start, graph = (x.resolve() for x in (args.rom, args.start_state, args.graph))
    eval_states = [('training_save', start)] + [(f'unseen_{i}', p.resolve()) for i, p in enumerate(args.eval_state, 1)]
    if any(sha256(path) == sha256(start) for _, path in eval_states[1:]):
        raise ValueError('Unseen evaluation saves must differ from the training save')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    potential_start = time.perf_counter()
    potential = make_potential(kind, graph, rom, 'Viridian City')
    potential_seconds = time.perf_counter() - potential_start
    lookups = load_lookups(ROOT / 'pokemon_rag/data/pret-pokered/processed/lookups.json')
    target_map = lookups['maps']['VIRIDIAN_CITY']['id']
    if kind == 'tile' and potential.target_map != target_map:
        raise ValueError('Graph target does not match the fixed experiment goal')
    rewards = {key: modified for key in ('level', 'opponent_level', 'death_penalty')}
    config = {'headless': True, 'save_final_state': False, 'early_stop': False, 'action_freq': 24,
        'init_state': str(start), 'max_steps': args.max_steps, 'print_rewards': False, 'save_video': False,
        'fast_video': True, 'session_path': output / 'sessions', 'gb_path': str(rom), 'debug': False,
        'reward_scale': 0.5, 'explore_weight': 0.25, 'rewards': rewards}
    config['session_path'].mkdir()
    metadata = {'condition': args.condition, 'beta': beta, 'potential': kind, 'rewards': rewards,
        'seed': args.seed, 'workers': args.workers, 'rollout_steps': args.n_steps,
        'rollouts': args.rollouts, 'training_timesteps': args.workers * args.n_steps * args.rollouts,
        'batch_size': args.batch_size, 'epochs': 1, 'gamma': args.gamma, 'learning_rate': 0.0003,
        'ent_coef': 0.01, 'max_episode_steps': args.max_steps, 'goal': 'Viridian City',
        'termination': 'goal map outside battle; otherwise time limit truncation',
        'terminal_potential': 0, 'unknown_potential': 0, 'evaluation': 'deterministic policy, one episode per save',
        'eval_every_rollouts': args.eval_every, 'rom_sha256': sha256(rom), 'start_sha256': sha256(start),
        'eval_sha256': {label: sha256(path) for label, path in eval_states},
        'graph_sha256': sha256(graph) if kind == 'tile' else None,
        'graph_construction_seconds': args.graph_discovery_seconds if kind == 'tile' else None,
        'graph_construction_cost_source': 'user supplied, not measured in this run',
        'potential_initialization_seconds': potential_seconds, 'device': 'cpu',
        'policy': 'stable-baselines3 MultiInputPolicy defaults',
        'map_lookups_sha256': sha256(ROOT / 'pokemon_rag/data/pret-pokered/processed/lookups.json'),
        'map_graph_sha256': sha256(ROOT / 'pokemon_rag/data/pret-pokered/processed/map_graph.json'),
        'source_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in (
            ROOT / 'pokemon_game/ppo_navigation_experiment.py', ROOT / 'pokemon_game/reward_shaping.py',
            ROOT / 'pokemon_game/value_function.py', ROOT / 'v2/red_gym_env_v2.py')},
        'versions': {'torch': torch.__version__, 'numpy': np.__version__,
                     'stable_baselines3': stable_baselines3.__version__}}
    (output / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')
    # RedGymEnv loads v2/events.json relative to the process working directory.
    sys.path.insert(0, str(ROOT / 'v2'))
    os.chdir(ROOT / 'v2')
    env = None
    eval_envs = []
    try:
        env = SubprocVecEnv([make_env(config, potential, beta, args.gamma, target_map)
                             for _ in range(args.workers)], start_method='forkserver')
        for label, path in eval_states:
            eval_config = {**config, 'init_state': str(path)}
            eval_envs.append((label, make_env(eval_config, MapPotential({i: 0.0 for i in range(256)}),
                                              0.0, args.gamma, target_map)()))
        model = PPO('MultiInputPolicy', env, n_steps=args.n_steps, batch_size=args.batch_size,
                    n_epochs=1, gamma=args.gamma, learning_rate=0.0003, ent_coef=0.01,
                    seed=args.seed, device='cpu', verbose=0)
        recorder = Recorder(output, eval_envs, args.max_steps, args.eval_every, started)
        model.learn(total_timesteps=metadata['training_timesteps'], callback=recorder)
        model.save(str(output / 'final_model'))
        (output / 'complete.json').write_text(json.dumps({'timesteps': model.num_timesteps,
            'wall_seconds': time.perf_counter() - started}, indent=2) + '\n')
        print(f"Completed {args.condition} seed {args.seed}: {model.num_timesteps} transitions", flush=True)
    finally:
        for _, eval_env in eval_envs:
            eval_env.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--rom', type=Path, default=ROOT / 'PokemonRed.gb')
    parser.add_argument('--start-state', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--eval-state', type=Path, action='append', default=[])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--n-steps', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--rollouts', type=int, default=61)
    parser.add_argument('--gamma', type=float, default=0.997)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--graph-discovery-seconds', type=float)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
