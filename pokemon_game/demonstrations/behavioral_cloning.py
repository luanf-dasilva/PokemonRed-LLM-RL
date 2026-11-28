"""Behavioral cloning for SB3 multi-input PPO policies."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import is_image_space

from pokemon_game.demonstrations.dataset import DemonstrationDataset
from pokemon_game.demonstrations.environment import ROOT, enter_v2, make_env


def validate_observations(model, dataset):
    spaces = model.observation_space.spaces
    if set(spaces) != set(dataset.schema):
        raise ValueError('Demonstration keys differ from policy observation space')
    for key, space in spaces.items():
        observed = tuple(dataset.schema[key][0])
        compatible_image = (is_image_space(space) and len(observed) == 3 and
                            (observed[2], observed[0], observed[1]) == tuple(space.shape))
        if tuple(space.shape) != observed and not compatible_image:
            raise ValueError(f'Observation shape mismatch for {key}')


def accuracy(model, dataset, batch_size=512):
    validate_observations(model, dataset)
    model.policy.set_training_mode(False)
    correct = total = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            obs, actions = dataset.batch(range(start, min(start + batch_size, len(dataset))))
            predicted, _ = model.predict(obs, deterministic=True)
            correct += int(np.count_nonzero(np.asarray(predicted).reshape(-1) == actions))
            total += len(actions)
    return correct / total


def clone(model, dataset, *, epochs=10, batch_size=512, learning_rate=1e-4, seed=0):
    """Optimize supervised action likelihood with an optimizer separate from PPO."""
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Invalid behavioral cloning settings')
    validate_observations(model, dataset)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    losses = []
    model.policy.set_training_mode(True)
    for _ in range(epochs):
        order = rng.permutation(len(dataset))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            obs, actions = dataset.batch(indices)
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            action_tensor = torch.as_tensor(actions, dtype=torch.long, device=model.device)
            _, log_prob, _ = model.policy.evaluate_actions(obs_tensor, action_tensor)
            loss = -log_prob.mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite cloning loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        losses.append(total_loss / len(dataset))
    model.policy.set_training_mode(False)
    return losses


def validate_split(train, validation):
    if validation is None:
        return
    if train.goals != validation.goals:
        raise ValueError('Training and validation goals differ')
    if any(not session['start_sha256'] for session in train.sessions + validation.sessions):
        raise ValueError('Training and validation sessions require starting snapshot hashes')
    train_ids = {s['start_sha256'] for s in train.sessions if s['start_sha256']}
    val_ids = {s['start_sha256'] for s in validation.sessions if s['start_sha256']}
    if train_ids & val_ids:
        raise ValueError('Training and validation sessions share an exact starting snapshot')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', nargs='+', type=Path, required=True)
    parser.add_argument('--validation-sessions', nargs='*', type=Path, default=[])
    parser.add_argument('--sources', nargs='+', default=['human', 'greedy_expert'])
    parser.add_argument('--allow-mixed-goals', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--rom', type=Path, default=ROOT / 'PokemonRed.gb')
    parser.add_argument('--start-state', type=Path, default=ROOT / 'init.state')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    rom, start_state = args.rom.resolve(), args.start_state.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    train = DemonstrationDataset(args.sessions, args.sources, allow_mixed_goals=args.allow_mixed_goals)
    validation = (DemonstrationDataset(args.validation_sessions, args.sources,
                                        allow_mixed_goals=args.allow_mixed_goals)
                  if args.validation_sessions else None)
    validate_split(train, validation)
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    env = None
    try:
        if args.checkpoint:
            model = PPO.load(str(args.checkpoint.resolve()), device='cpu')
        else:
            enter_v2()
            env = make_env(rom, start_state, output / 'sessions')
            model = PPO('MultiInputPolicy', env, n_steps=16, batch_size=16, seed=args.seed,
                        device='cpu', verbose=0)
        before = accuracy(model, train, args.batch_size)
        val_before = accuracy(model, validation, args.batch_size) if validation else None
        losses = clone(model, train, epochs=args.epochs, batch_size=args.batch_size,
                       learning_rate=args.learning_rate, seed=args.seed)
        metrics = {'train_sessions': train.sessions, 'validation_sessions': validation.sessions if validation else [],
            'sources': args.sources, 'goals': list(train.goals), 'allow_mixed_goals': args.allow_mixed_goals,
            'train_transitions': len(train), 'validation_transitions': len(validation) if validation else 0,
            'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
            'seed': args.seed, 'losses': losses, 'train_accuracy_before': before,
            'train_accuracy_after': accuracy(model, train, args.batch_size),
            'validation_accuracy_before': val_before,
            'validation_accuracy_after': accuracy(model, validation, args.batch_size) if validation else None,
            'checkpoint': str(args.checkpoint.resolve()) if args.checkpoint else None}
        model.save(str(output / 'policy'))
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps(metrics, indent=2))
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
