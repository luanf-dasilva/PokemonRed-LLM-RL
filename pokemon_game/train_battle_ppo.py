"""Train a separate semantic-action PPO battle policy; never runs on import."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from pokemon_game.battle_env import PokemonBattleEnv


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value) + '\n')


def evaluate(model, env, output, step, started):
    for scenario in range(len(env.scenarios)):
        obs, _ = env.reset(options={'scenario': scenario})
        total_reward, actions = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            actions += 1
            if terminated or truncated:
                break
        append(output / 'evaluation.jsonl', {'training_abstract_actions': step, 'scenario': scenario,
            'outcome': info['outcome'], 'terminated': terminated, 'truncated': truncated,
            'abstract_actions': actions, 'emulator_frames': env.controller.frames,
            'controller_inputs': env.controller.inputs, 'return': total_reward,
            'elapsed_wall_seconds': time.perf_counter() - started})


class Recorder(BaseCallback):
    def __init__(self, output, eval_env, interval, started):
        super().__init__()
        self.output, self.eval_env, self.interval, self.started = output, eval_env, interval, started
        self.rollouts = self.frames = self.inputs = self.invalid = 0

    def _on_training_start(self):
        evaluate(self.model, self.eval_env, self.output, 0, self.started)

    def _on_rollout_start(self):
        if self.rollouts and self.rollouts % self.interval == 0:
            evaluate(self.model, self.eval_env, self.output, self.num_timesteps, self.started)

    def _on_step(self):
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            self.frames += info['emulator_frames']
            self.inputs += info['controller_inputs']
            self.invalid += int(info['outcome'] == 'invalid_move')
            if done:
                append(self.output / 'episodes.jsonl', {'training_abstract_actions': self.num_timesteps,
                    'outcome': info['outcome'], 'cumulative_emulator_frames': self.frames,
                    'cumulative_controller_inputs': self.inputs, 'cumulative_invalid_actions': self.invalid,
                    'elapsed_wall_seconds': time.perf_counter() - self.started})
        return True

    def _on_rollout_end(self):
        self.rollouts += 1
        append(self.output / 'training.jsonl', {'rollout': self.rollouts,
            'training_abstract_actions': self.num_timesteps, 'emulator_frames': self.frames,
            'controller_inputs': self.inputs, 'invalid_actions': self.invalid,
            'elapsed_wall_seconds': time.perf_counter() - self.started})

    def _on_training_end(self):
        evaluate(self.model, self.eval_env, self.output, self.num_timesteps, self.started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rom', required=True, type=Path)
    parser.add_argument('--train-archive', required=True, type=Path)
    parser.add_argument('--eval-archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rollout-steps', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--rollouts', type=int, default=40)
    parser.add_argument('--eval-every', type=int, default=5)
    args = parser.parse_args()
    if (args.rollout_steps < 2 or args.batch_size < 2 or args.rollout_steps % args.batch_size
            or args.rollouts < 1 or args.eval_every < 1):
        raise ValueError('Invalid rollout, batch, or evaluation settings')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    train = PokemonBattleEnv(args.rom, args.train_archive)
    eval_env = None
    try:
        eval_env = PokemonBattleEnv(args.rom, args.eval_archive)
        output.mkdir(parents=True)
        started = time.perf_counter()
        protocol = {'policy': 'new PPO MlpPolicy', 'seed': args.seed, 'rollout_steps': args.rollout_steps,
            'batch_size': args.batch_size, 'rollouts': args.rollouts, 'eval_every_rollouts': args.eval_every,
            'training_abstract_actions': args.rollout_steps * args.rollouts, 'gamma': 0.99,
            'rom_sha256': sha(args.rom), 'train_manifest_sha256': sha(args.train_archive / 'manifest.json'),
            'eval_manifest_sha256': sha(args.eval_archive / 'manifest.json'),
            'invalid_actions': 'unmasked PPO; invalid move returns a penalty and consumes one abstract action',
            'discounting': 'gamma per abstract action, regardless of number of emulator frames',
            'evaluation': 'deterministic policy, once per fixed held-out snapshot'}
        (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        model = PPO('MlpPolicy', train, n_steps=args.rollout_steps, batch_size=args.batch_size,
                    gamma=0.99, seed=args.seed, device='cpu', verbose=0)
        model.learn(total_timesteps=protocol['training_abstract_actions'],
                    callback=Recorder(output, eval_env, args.eval_every, started))
        model.save(str(output / 'final_model'))
        (output / 'complete.json').write_text(json.dumps({'timesteps': model.num_timesteps,
            'wall_seconds': time.perf_counter() - started}, indent=2) + '\n')
    finally:
        if eval_env is not None:
            eval_env.close()
        train.close()


if __name__ == '__main__':
    main()
