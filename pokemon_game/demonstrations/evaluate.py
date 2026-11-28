"""Evaluate a cloned PPO policy through Gym without the navigation graph."""
import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import PPO
import torch

from pokemon_game.demonstrations.environment import ROOT, enter_v2, make_env
from pokemon_game.emulator_simulator import BATTLE_FLAG
from pokemon_game.goal_conditioning import GOAL_KEY
from pokemon_game.state_encoder import MAP_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--start-state', type=Path, required=True)
    parser.add_argument('--rom', type=Path, default=ROOT / 'PokemonRed.gb')
    parser.add_argument('--goal-map-id', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=300)
    args = parser.parse_args()
    output, checkpoint, start, rom = (p.resolve() for p in
        (args.output, args.checkpoint, args.start_state, args.rom))
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    model = PPO.load(str(checkpoint), device='cpu')
    conditioned = GOAL_KEY in model.observation_space.spaces
    enter_v2()
    env = make_env(rom, start, output / 'sessions', args.max_steps,
                   target_map=None if conditioned else args.goal_map_id,
                   goal_maps=[args.goal_map_id] if conditioned else None)
    try:
        observation, _ = env.reset()
        started = time.perf_counter()
        reward_total, battle_entered, actions = 0.0, False, []
        outcome = 'step_limit'
        for _ in range(args.max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            actions.append(int(action))
            reward_total += float(reward)
            battle_entered |= bool(env.unwrapped.pyboy.memory[BATTLE_FLAG])
            if info.get('goal_reached'):
                outcome = 'goal_reached'
                break
            if terminated or truncated:
                outcome = 'environment_ended'
                break
        result = {'outcome': outcome, 'goal_map_id': args.goal_map_id, 'goal_conditioned': conditioned, 'actions': actions,
                  'actual_steps': len(actions), 'original_return': reward_total,
                  'final_map_id': int(env.unwrapped.pyboy.memory[MAP_ID]),
                  'battle_entered': battle_entered, 'simulated_actions': 0,
                  'wall_seconds': time.perf_counter() - started}
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: value for key, value in result.items() if key != 'actions'}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
