"""Record model-based navigation demonstrations through the Gym environment."""
import argparse
import hashlib
import json
from pathlib import Path

from pokemon_game.demonstrations.environment import ROOT, enter_v2, make_env
from pokemon_game.demonstrations.recorder import DemonstrationRecorder
from pokemon_game.emulator_simulator import EmulatorSimulator
from pokemon_game.greedy_policy import correct_transition, select_model_action
from pokemon_game.state_encoder import load_lookups
from pokemon_game.tile_graph import distances_to_target, load_graph, position_key


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(rom, start_state, graph_path, output, max_steps=300, allow_compatible_start=False):
    rom, start_state, graph_path, output = map(lambda p: Path(p).resolve(), (rom, start_state, graph_path, output))
    data = json.loads(graph_path.read_text())
    context = data['context']
    if (context['rom_sha256'] != digest(rom) or context['action_timing'] != [8, 16, 0] or
            context['directions'] != ['down', 'left', 'right', 'up']):
        raise ValueError('Graph ROM or action timing is incompatible')
    graph = load_graph(graph_path, context)
    if graph.target is None or graph.target[0] != context['target_map']:
        raise ValueError('Graph target is incompatible')
    lookups = load_lookups(ROOT / 'pokemon_rag/data/pret-pokered/processed/lookups.json')
    enter_v2()
    env = make_env(rom, start_state, output.parent / (output.name + '-sessions'), max_steps,
                   target_map=context['target_map'])
    recorder = None
    try:
        recorder = DemonstrationRecorder(env, output, lookups, 'greedy_expert',
                                         rom_sha256=digest(rom), start_sha256=digest(start_state))
        recorder.manifest.update({'goal_map_id': context['target_map'], 'graph_sha256': digest(graph_path),
                                  'simulated_actions': 0, 'prediction_mismatches': 0})
        recorder.reset()
        simulator = EmulatorSimulator(env.unwrapped.pyboy)
        exact = hashlib.sha256(simulator.snapshot()).hexdigest() == context['start_sha256']
        recorder.manifest['exact_graph_start'] = exact
        if not exact and not allow_compatible_start:
            raise ValueError('Graph was discovered from a different emulator snapshot')
        if exact and position_key(simulator.observe()) != graph.start:
            raise ValueError('Graph start position does not match')
        distances = distances_to_target(graph)
        if position_key(simulator.observe()) not in distances:
            raise ValueError('Starting position has no discovered path to the target')
        outcome = 'step_limit'
        for _ in range(max_steps):
            current = simulator.observe()
            if current.in_battle:
                outcome = 'battle'
                break
            if current.map_id == context['target_map']:
                outcome = 'goal_reached'
                break
            try:
                action = select_model_action(graph, current, distances)
            except RuntimeError:
                outcome = 'no_known_path'
                break
            source = position_key(current)
            predicted = graph.edges[source][action]
            _, _, terminated, truncated, info = recorder.step(action)
            actual = simulator.observe()
            if correct_transition(graph, source, action, predicted, actual, set(context['allowed_maps'])):
                recorder.manifest['prediction_mismatches'] += 1
                distances = distances_to_target(graph)
            if info.get('goal_reached'):
                outcome = 'goal_reached'
                break
            if actual.in_battle:
                outcome = 'battle'
                break
            if terminated or truncated:
                outcome = 'environment_ended'
                break
        recorder.finish(outcome)
        print(json.dumps({'outcome': outcome, 'transitions': recorder.manifest['transitions'],
                          'prediction_mismatches': recorder.manifest['prediction_mismatches']}))
        return outcome
    finally:
        if recorder is not None:
            recorder.close()
        else:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rom', type=Path, default=ROOT / 'PokemonRed.gb')
    parser.add_argument('--start-state', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--allow-compatible-start', action='store_true',
                        help='Allow a different snapshot at a graph-reachable coordinate; validate live edges')
    args = parser.parse_args()
    run(args.rom, args.start_state, args.graph, args.output, args.max_steps, args.allow_compatible_start)


if __name__ == '__main__':
    main()
