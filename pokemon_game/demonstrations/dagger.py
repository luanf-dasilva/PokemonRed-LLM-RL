"""Aggregate expert labels on learner-visited states using isolated emulator probes."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from pyboy import PyBoy
from stable_baselines3 import PPO

from pokemon_game.demonstrations.behavioral_cloning import accuracy, clone
from pokemon_game.demonstrations.dataset import DemonstrationDataset
from pokemon_game.demonstrations.environment import ROOT, enter_v2, make_env
from pokemon_game.demonstrations.recorder import DemonstrationRecorder
from pokemon_game.emulator_simulator import EmulatorSimulator
from pokemon_game.goal_conditioning import GOAL_KEY
from pokemon_game.greedy_policy import correct_transition, select_model_action
from pokemon_game.state_encoder import load_lookups
from pokemon_game.tile_graph import discover, distances_to_target, load_graph, position_key


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ExpertOracle:
    """Validate proposed expert edges on a separate emulator before labeling."""
    def __init__(self, probe, graph, allowed, target, max_nodes, max_discoveries, recovery_horizon=6):
        self.probe, self.graph, self.allowed, self.target = probe, graph, set(allowed), target
        self.max_nodes, self.max_discoveries = max_nodes, max_discoveries
        self.discoveries = self.mismatches = 0
        self.recovery_horizon, self.recovery_labels = recovery_horizon, 0

    def _recovery_label(self, snapshot, source, distances):
        """Find a short action sequence returning to a verified graph transition."""
        choices = []
        for action in range(4):
            self.probe.restore(snapshot)
            for steps in range(1, self.recovery_horizon + 1):
                observed = self.probe.step(action, simulated=True)
                if observed.in_battle:
                    break
                key = position_key(observed)
                distance = distances.get(key)
                if distance is None or key == source:
                    continue
                if key[0] == self.target:
                    choices.append((steps + distance, action))
                    break
                # Warp animations can briefly report incoherent coordinates.
                # Verify an improving graph edge from this candidate before using it.
                candidate_snapshot = self.probe.snapshot()
                try:
                    next_action = select_model_action(self.graph, observed, distances)
                    predicted = self.graph.edges[key][next_action]
                    checked = self.probe.step(next_action, simulated=True)
                    if not checked.in_battle and position_key(checked) == predicted:
                        choices.append((steps + distance, action))
                        break
                except RuntimeError:
                    pass
                finally:
                    self.probe.restore(candidate_snapshot)
        if not choices:
            return None
        self.recovery_labels += 1
        return min(choices)[1]

    def label(self, snapshot, current):
        for _ in range(8):
            distances = distances_to_target(self.graph)
            try:
                action = select_model_action(self.graph, current, distances)
            except RuntimeError:
                recovery = self._recovery_label(snapshot, position_key(current), distances)
                if recovery is not None:
                    return recovery
                if self.discoveries >= self.max_discoveries:
                    return None
                self.probe.restore(snapshot)
                allowed = self.allowed | {current.map_id}
                refreshed = discover(self.probe, self.target, allowed, max_nodes=self.max_nodes)
                self.discoveries += 1
                if refreshed.target is None:
                    return None
                self.allowed |= allowed
                self.graph.edges.update(refreshed.edges)
                self.graph.target = refreshed.target
                continue
            source = position_key(current)
            predicted = self.graph.edges[source][action]
            self.probe.restore(snapshot)
            observed = self.probe.step(action, simulated=True)
            if not correct_transition(self.graph, source, action, predicted, observed, self.allowed):
                return action
            self.mismatches += 1
        return None


def run(args):
    output, rom, start, graph_path, checkpoint = (p.resolve() for p in
        (args.output, args.rom, args.start_state, args.graph, args.checkpoint))
    initial_sessions = [p.resolve() for p in args.initial_sessions]
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    if args.rounds < 1 or args.max_steps < 1 or args.max_nodes < 1 or args.max_discoveries < 0 or args.recovery_horizon < 0:
        raise ValueError('Invalid DAgger limits')
    if not 0 <= args.expert_probability <= 1:
        raise ValueError('Expert probability must be between 0 and 1')
    graph_data = json.loads(graph_path.read_text())
    context = graph_data['context']
    if (context['rom_sha256'] != sha(rom) or context['action_timing'] != [8, 16, 0] or
            context['directions'] != ['down', 'left', 'right', 'up']):
        raise ValueError('Graph ROM or action timing is incompatible')
    if args.allowed_map_ids and any(not 0 <= value <= 255 for value in args.allowed_map_ids):
        raise ValueError('Allowed map IDs must be bytes')
    target = context['target_map']
    lookups = load_lookups(ROOT / 'pokemon_rag/data/pret-pokered/processed/lookups.json')
    model = PPO.load(str(checkpoint), device='cpu')
    conditioned = GOAL_KEY in model.observation_space.spaces
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    protocol = {'checkpoint_sha256': sha(checkpoint), 'rom_sha256': sha(rom),
        'start_sha256': sha(start), 'graph_sha256': sha(graph_path), 'goal_map_id': target,
        'goal_conditioned': conditioned, 'rounds': args.rounds, 'max_steps': args.max_steps,
        'expert_probability': args.expert_probability, 'seed': args.seed,
        'max_nodes_per_discovery': args.max_nodes, 'max_discoveries_per_round': args.max_discoveries,
        'recovery_horizon': args.recovery_horizon,
        'initial_sessions': [str(p) for p in initial_sessions],
        'expert': 'one-step validation; bounded repeated-direction recovery and discovery on missing paths',
        'semantics': 'observation and successor follow behavior action; supervised label is expert action',
        'limitation': 'expert probes sample current emulator state and do not guarantee future validity'}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    enter_v2()
    rng = random.Random(args.seed)
    sessions = list(initial_sessions)
    for round_number in range(1, args.rounds + 1):
        path = output / f'dagger-round-{round_number:03d}'
        env = make_env(rom, start, output / f'sessions-{round_number:03d}', args.max_steps,
                       target_map=None if conditioned else target,
                       goal_maps=[target] if conditioned else None)
        probe_pyboy = None
        recorder = None
        try:
            recorder = DemonstrationRecorder(env, path, lookups, 'dagger_expert',
                                             rom_sha256=sha(rom), start_sha256=sha(start))
            recorder.manifest.update({'goal_map_id': target, 'graph_sha256': sha(graph_path),
                                      'round': round_number, 'goal_conditioned': conditioned})
            obs, _ = recorder.reset()
            live = EmulatorSimulator(env.unwrapped.pyboy)
            probe_pyboy = PyBoy(str(rom), window='null')
            probe_pyboy.set_emulation_speed(0)
            oracle = ExpertOracle(EmulatorSimulator(probe_pyboy), load_graph(graph_path, context),
                                  set(context['allowed_maps']) | set(args.allowed_map_ids or []),
                                  target, args.max_nodes, args.max_discoveries, args.recovery_horizon)
            outcome, disagreements, learner_actions, expert_actions = 'step_limit', 0, 0, 0
            for _ in range(args.max_steps):
                current = live.observe()
                if current.in_battle:
                    outcome = 'battle'
                    break
                if current.map_id == target:
                    outcome = 'goal_reached'
                    break
                live_snapshot = live.snapshot()
                label = oracle.label(live_snapshot, current)
                if live.snapshot() != live_snapshot:
                    raise RuntimeError('Expert probing modified the live emulator')
                if label is None:
                    outcome = 'expert_abstained'
                    break
                prediction, _ = model.predict(obs, deterministic=True)
                predicted_action = int(prediction)
                disagreements += int(predicted_action != label)
                use_expert = rng.random() < args.expert_probability
                behavior = label if use_expert else predicted_action
                learner_actions += int(not use_expert)
                expert_actions += int(use_expert)
                obs, _, terminated, truncated, info = recorder.step_labeled(
                    behavior, label, 'expert' if use_expert else 'learner')
                if info.get('goal_reached'):
                    outcome = 'goal_reached'
                    break
                if live.observe().in_battle:
                    outcome = 'battle'
                    break
                if terminated or truncated:
                    outcome = 'environment_ended'
                    break
            recorder.manifest.update({'expert_simulated_actions': oracle.probe.simulated_actions,
                'expert_simulated_frames': oracle.probe.simulated_frames, 'discoveries': oracle.discoveries,
                'prediction_mismatches': oracle.mismatches, 'recovery_labels': oracle.recovery_labels,
                'expert_learner_disagreements': disagreements,
                'learner_behavior_actions': learner_actions, 'expert_behavior_actions': expert_actions})
            recorder.finish(outcome)
            count = recorder.manifest['transitions']
            record = {'round': round_number, 'outcome': outcome, 'labeled_transitions': count,
                      'expert_simulated_actions': oracle.probe.simulated_actions,
                      'expert_simulated_frames': oracle.probe.simulated_frames, 'discoveries': oracle.discoveries,
                      'prediction_mismatches': oracle.mismatches, 'disagreements': disagreements,
                      'recovery_labels': oracle.recovery_labels,
                      'learner_behavior_actions': learner_actions, 'expert_behavior_actions': expert_actions}
        finally:
            if probe_pyboy is not None:
                probe_pyboy.stop(save=False)
            if recorder is not None:
                recorder.close()
            else:
                env.close()
        if count:
            sessions.append(path)
        if not sessions:
            record['update'] = 'no_labeled_data'
            with (output / 'rounds.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            break
        dataset = DemonstrationDataset(sessions, goal_conditioned=conditioned)
        if dataset.goals != {target}:
            raise ValueError('All DAgger data must match the graph target')
        record['aggregate_transitions'] = len(dataset)
        record['action_accuracy_before'] = accuracy(model, dataset, args.batch_size)
        record['losses'] = clone(model, dataset, epochs=args.epochs, batch_size=args.batch_size,
                                 learning_rate=args.learning_rate, seed=args.seed + round_number)
        record['action_accuracy_after'] = accuracy(model, dataset, args.batch_size)
        model.save(str(output / f'model-round-{round_number:03d}'))
        with (output / 'rounds.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rom', type=Path, default=ROOT / 'PokemonRed.gb')
    parser.add_argument('--start-state', required=True, type=Path)
    parser.add_argument('--graph', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--initial-sessions', nargs='*', type=Path, default=[])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--max-nodes', type=int, default=1000)
    parser.add_argument('--max-discoveries', type=int, default=2)
    parser.add_argument('--recovery-horizon', type=int, default=6)
    parser.add_argument('--allowed-map-ids', nargs='*', type=int, default=[])
    parser.add_argument('--expert-probability', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=0)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
