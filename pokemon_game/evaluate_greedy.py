"""Compare navigation policies from the same post-starter emulator state."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from pokemon_game.emulator_simulator import EmulatorSimulator
from pokemon_game.greedy_policy import correct_transition, select_action, select_model_action, tile_value
from pokemon_game.goal_resolver import plan_goal, resolve_goal
from pokemon_game.state_encoder import encode_state, load_lookups
from pokemon_game.tile_graph import discover, distances_to_target, position_key, load_graph, save_graph
from pokemon_game.value_function import HierarchicalNavigationValue, LocationValue

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROM = ROOT / "PokemonRed.gb"
DEFAULT_SOURCE = ROOT / "has_pokedex.state"
DEFAULT_CHECKPOINT = ROOT / "v2/runs/poke_125829120_steps.zip"
POLICIES = ("random", "map_greedy", "tile_greedy", "hierarchical_greedy", "cached_greedy", "model_greedy", "ppo")
DEFAULT_POLICIES = ("random", "map_greedy", "tile_greedy", "hierarchical_greedy", "ppo")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_start(rom, source, destination, lookups):
    """Exit Oak's Lab using a post-starter save; leave source files unchanged."""
    from pyboy import PyBoy
    pyboy = PyBoy(str(rom), window="null")
    pyboy.set_emulation_speed(0)
    try:
        with open(source, "rb") as stream:
            pyboy.load_state(stream)
        sim = EmulatorSimulator(pyboy)
        for _ in range(20):
            if sim.observe().map_id == lookups["maps"]["PALLET_TOWN"]["id"]:
                break
            if sim.observe().map_id != lookups["maps"]["OAKS_LAB"]["id"]:
                raise ValueError("Source save is not in Oak's Lab or Pallet Town")
            sim.step(0)
        # Finish the transition animation before taking the shared starting state.
        pyboy.tick(48, True)
        state = encode_state(pyboy.memory, lookups)
        if state["map_name"] != "PALLET_TOWN" or not state["has_starter"]:
            raise ValueError("Could not prepare a post-starter Pallet Town state")
        with open(destination, "wb") as stream:
            pyboy.save_state(stream)
        return {"state": state, "preparation_actions": sim.committed_actions,
                "preparation_frames": sim.committed_frames + 48}
    finally:
        pyboy.stop(save=False)


def run_episode(policy, seed, start_path, rom, model, lookups, map_edges, max_steps, output, graph_limit,
                goal=None, allowed=None, cache_dir=None):
    from pyboy import PyBoy
    from red_gym_env_v2 import RedGymEnv

    start_time = time.perf_counter()
    config = {"headless": True, "save_final_state": False, "early_stop": False,
        "action_freq": 24, "init_state": str(start_path), "max_steps": max_steps + 1,
        "print_rewards": False, "save_video": False, "fast_video": True,
        "session_path": output / "sessions" / f"{policy}-{seed}", "gb_path": str(rom),
        "debug": False, "reward_scale": 0.5, "explore_weight": 0.25}
    config["session_path"].mkdir(parents=True, exist_ok=True)
    env = RedGymEnv(config)
    env.pyboy.set_emulation_speed(0)
    probe = None
    try:
        observation, _ = env.reset(seed=seed)
        live = EmulatorSimulator(env.pyboy)
        start_snapshot = live.snapshot()
        rng = random.Random(seed)
        maps_by_id = {v["id"]: name for name, v in lookups["maps"].items()}
        map_value = LocationValue(lookups, map_edges)
        if allowed is None:
            allowed = {lookups["maps"][name]["id"] for name in ("PALLET_TOWN", "ROUTE_1", "VIRIDIAN_CITY")}
        goal = goal or {"goal_type": "REACH_LOCATION", "target": "Viridian City"}
        target_id = resolve_goal(goal, lookups)
        if target_id not in allowed:
            raise ValueError("Target must be included in allowed maps")
        cache_context = {"rom_sha256": digest(rom), "start_sha256": hashlib.sha256(start_snapshot).hexdigest(),
                         "target_map": target_id, "allowed_maps": sorted(allowed), "graph_limit": graph_limit,
                         "action_timing": [8, 16, 0], "directions": ["down", "left", "right", "up"]}
        cache_path = None
        if cache_dir is not None:
            cache_key = hashlib.sha256(json.dumps(cache_context, sort_keys=True).encode()).hexdigest()
            cache_path = Path(cache_dir) / (cache_key + ".json")
        cache_hit, mismatches, repairs = False, 0, 0
        discovery_info = None
        if policy in {"map_greedy", "tile_greedy", "hierarchical_greedy", "cached_greedy", "model_greedy"}:
            probe_pyboy = PyBoy(str(rom), window="null")
            probe_pyboy.set_emulation_speed(0)
            probe = EmulatorSimulator(probe_pyboy)
            probe.restore(start_snapshot)
        planning_start = time.perf_counter()
        distances = None
        hierarchical = None
        if policy in {"tile_greedy", "hierarchical_greedy", "cached_greedy", "model_greedy"}:
            before_count = env.step_count
            before_seen = dict(env.seen_coords)
            if policy in {"cached_greedy", "model_greedy"} and cache_path is not None and cache_path.exists():
                discovered = load_graph(cache_path, cache_context)
                cache_hit = True
            else:
                discovered = discover(probe, target_id, allowed, max_nodes=graph_limit)
                if policy in {"cached_greedy", "model_greedy"} and cache_path is not None and discovered.target is not None:
                    save_graph(cache_path, discovered, cache_context)
            if live.snapshot() != start_snapshot or env.step_count != before_count or env.seen_coords != before_seen:
                raise RuntimeError("Speculative discovery modified the live emulator or environment")
            if discovered.target is None:
                planning_seconds = time.perf_counter() - planning_start
                return {"policy": policy, "seed": seed, "outcome": "no_path_discovered",
                    "goal_reached": False, "actual_environment_steps": 0, "actual_emulator_frames": 0,
                    "simulated_actions": probe.simulated_actions, "simulated_frames": probe.simulated_frames,
                    "planning_seconds": planning_seconds, "execution_seconds": 0,
                    "wall_seconds": time.perf_counter() - start_time, "unique_coordinates": 1,
                    "repeated_states": 0, "battle_entered": False, "battle_steps": 0,
                    "final_position": position_key(live.observe()), "actions": [],
                    "discovery": {"nodes": discovered.explored_nodes,
                        "edge_count": sum(map(len, discovered.edges.values())), "truncated": discovered.truncated,
                        "start_distance": None, "planning_simulated_actions": probe.simulated_actions,
                        "planning_simulated_frames": probe.simulated_frames}}
            distances = distances_to_target(discovered)
            if policy == "hierarchical_greedy":
                hierarchical = HierarchicalNavigationValue(discovered)
            discovery_info = {"nodes": discovered.explored_nodes, "edge_count": sum(map(len, discovered.edges.values())),
                "truncated": discovered.truncated, "start_distance": distances.get(discovered.start),
                "planning_simulated_actions": probe.simulated_actions, "planning_simulated_frames": probe.simulated_frames}
            graph_file = output / f"{policy}-graph-{seed}.json"
            graph_file.write_text(json.dumps({"start": discovered.start, "target": discovered.target,
                "edges": [{"from": key, "actions": {str(a): dest for a, dest in edges.items()}}
                          for key, edges in discovered.edges.items()]}, indent=2) + "\n")
        planning_seconds = time.perf_counter() - planning_start
        unique = {position_key(live.observe())}
        repeated = 0
        actions = []
        outcome = "step_limit"
        battle_steps = 0
        battle_entered = False
        start_execution = time.perf_counter()
        for _ in range(max_steps):
            current = live.observe()
            if current.in_battle and policy in {"map_greedy", "tile_greedy", "hierarchical_greedy", "cached_greedy", "model_greedy"}:
                outcome = "battle"
                break
            if current.map_id == target_id:
                outcome = "goal_reached"
                break
            if current.in_battle:
                battle_steps += 1
            if policy == "random":
                action = rng.randrange(env.action_space.n)
            elif policy == "ppo":
                action, _ = model.predict(observation, deterministic=True)
                action = int(action)
            else:
                probe.restore(live.snapshot())
                if policy in {"tile_greedy", "cached_greedy", "model_greedy"}:
                    score = lambda after: tile_value(distances, after)
                elif policy == "hierarchical_greedy":
                    score = hierarchical.evaluate
                else:
                    def score(after):
                        if after.map_id not in allowed:
                            return None
                        try:
                            return map_value.evaluate({"map_name": maps_by_id[after.map_id]}, goal)
                        except (ValueError, KeyError):
                            return None
                try:
                    if policy == "model_greedy":
                        try:
                            action = select_model_action(discovered, current, distances)
                        except RuntimeError:
                            # Discover from the actual state only when the model has no usable path.
                            refreshed = discover(probe, target_id, allowed, max_nodes=graph_limit)
                            repairs += 1
                            if refreshed.target is None:
                                raise RuntimeError("No path discovered from actual state")
                            discovered.edges.update(refreshed.edges)
                            discovered.target = refreshed.target
                            distances = distances_to_target(discovered)
                            if cache_path is not None:
                                save_graph(cache_path, discovered, cache_context)
                            action = select_model_action(discovered, current, distances)
                    else:
                        action = select_action(probe, score)
                except RuntimeError:
                    outcome = "no_eligible_action"
                    break
            previous = position_key(current)
            predicted = discovered.edges.get(previous, {}).get(action) if policy == "model_greedy" else None
            observation, _, terminated, truncated, _ = env.step(action)
            actions.append(action)
            current = live.observe()
            battle_entered = battle_entered or current.in_battle
            key = position_key(current)
            if policy == "model_greedy" and correct_transition(discovered, previous, action, predicted, current, allowed):
                mismatches += 1
                distances = distances_to_target(discovered)
                if cache_path is not None:
                    save_graph(cache_path, discovered, cache_context)
            if key in unique:
                repeated += 1
            unique.add(key)
            if current.in_battle and policy in {"map_greedy", "tile_greedy", "hierarchical_greedy", "cached_greedy", "model_greedy"}:
                outcome = "battle"
                break
            if current.map_id == target_id:
                outcome = "goal_reached"
                break
            if terminated or truncated:
                outcome = "environment_terminated" if terminated else "environment_truncated"
                break
        execution_seconds = time.perf_counter() - start_execution
        final_state = output / f"{policy}-final-{seed}.state"
        final_state.write_bytes(live.snapshot())
        return {"policy": policy, "seed": seed, "outcome": outcome,
            "goal_reached": outcome == "goal_reached", "actual_environment_steps": len(actions),
            "actual_emulator_frames": len(actions) * 24,
            "simulated_actions": probe.simulated_actions if probe else 0,
            "simulated_frames": probe.simulated_frames if probe else 0,
            "planning_seconds": planning_seconds, "execution_seconds": execution_seconds,
            "wall_seconds": time.perf_counter() - start_time,
            "unique_coordinates": len(unique), "repeated_states": repeated,
            "battle_entered": battle_entered, "battle_steps": battle_steps,
            "final_position": position_key(live.observe()), "discovery": discovery_info, "actions": actions,
            "cache_hit": cache_hit, "prediction_mismatches": mismatches, "graph_repairs": repairs,
            "final_state_file": final_state.name}
    finally:
        if probe is not None:
            probe.pyboy.stop(save=False)
        env.pyboy.stop(save=False)


def _run(output, rom=DEFAULT_ROM, source=DEFAULT_SOURCE, checkpoint=DEFAULT_CHECKPOINT,
         episodes=3, max_steps=300, graph_limit=1000, policies=DEFAULT_POLICIES,
         cache_dir=None, target="Viridian City", allowed_maps=None, llm_objective=None, index_path=None,
         start_state=None):
    if episodes < 1 or max_steps < 1 or graph_limit < 1:
        raise ValueError("Episode, step and graph limits must be positive")
    if not policies or any(policy not in POLICIES for policy in policies) or len(set(policies)) != len(policies):
        raise ValueError("Invalid or duplicated policy")
    output, rom, source, checkpoint = map(lambda p: Path(p).resolve(), (output, rom, source, checkpoint))
    if start_state is not None:
        start_state = Path(start_state).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True)
    lookups = load_lookups(ROOT / "pokemon_rag/data/pret-pokered/processed/lookups.json")
    graph = json.loads((ROOT / "pokemon_rag/data/pret-pokered/processed/map_graph.json").read_text())
    started = time.perf_counter()
    start_path = output / "pallet-post-starter.state"
    if start_state is None:
        start_info = prepare_start(rom, source, start_path, lookups)
    else:
        from pyboy import PyBoy
        start_path.write_bytes(Path(start_state).read_bytes())
        pyboy = PyBoy(str(rom), window="null")
        try:
            with start_path.open("rb") as stream:
                pyboy.load_state(stream)
            start_info = {"state": encode_state(pyboy.memory, lookups),
                          "preparation_actions": 0, "preparation_frames": 0}
        finally:
            pyboy.stop(save=False)
    setup_seconds = time.perf_counter() - started
    goal = {"goal_type": "REACH_LOCATION", "target": target}
    if llm_objective:
        try:
            result = plan_goal(start_info["state"], llm_objective, index_path)
            (output / "planner.json").write_text(json.dumps(result, indent=2) + "\n")
            if not result["schema_valid"] or not result["citation_ids_valid"]:
                raise ValueError("Planner returned invalid schema or citation IDs")
            goal = result["goal"]
            if goal["goal_type"] != "INSUFFICIENT_CONTEXT" and not result["evidence_present"]:
                raise ValueError("Planner did not cite evidence")
            resolve_goal(goal, lookups)
        except Exception as exc:
            (output / "planner_error.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")
            raise
    target_id = resolve_goal(goal, lookups)
    names = allowed_maps or ["PALLET_TOWN", "ROUTE_1", "VIRIDIAN_CITY"]
    allowed = {resolve_goal({"goal_type": "REACH_LOCATION", "target": name}, lookups) for name in names}
    if target_id not in allowed:
        if llm_objective:
            (output / "planner_error.json").write_text(json.dumps(
                {"error": "Planner target is outside the allowed maps", "target": goal["target"]}, indent=2) + "\n")
        raise ValueError("Target must be included in --allowed-maps")
    if cache_dir is not None:
        cache_dir = Path(cache_dir).resolve()
    # v2 reads events.json relative to its working directory.
    sys.path.insert(0, str(ROOT / "v2"))
    os.chdir(ROOT / "v2")
    started = time.perf_counter()
    model = None
    if "ppo" in policies:
        from stable_baselines3 import PPO
        model = PPO.load(str(checkpoint), device="cpu")
    model_load_seconds = time.perf_counter() - started
    protocol = {"goal": goal, "llm_objective": llm_objective, "allowed_maps": names, "cache_dir": str(cache_dir) if cache_dir else None,
        "source_state_sha256": digest(source if start_state is None else start_state), "start_state_sha256": digest(start_path),
        "rom_sha256": digest(rom), "checkpoint_sha256": digest(checkpoint) if model else None,
        "checkpoint_name": checkpoint.name if model else None, "start": start_info, "setup_seconds": setup_seconds,
        "ppo_model_load_seconds": model_load_seconds, "episodes_per_policy": episodes,
        "max_steps": max_steps, "graph_limit": graph_limit, "frame_timing": "8 pressed + 16 released",
        "action_sets": {"random": "all 7 actions", "ppo": "all 7 actions, deterministic",
                        "map_greedy": "4 directional actions", "tile_greedy": "4 directional actions",
                        "hierarchical_greedy": "4 directional actions", "cached_greedy": "4 directional actions",
                        "model_greedy": "4 directional actions"},
        "policies": list(policies),
        "termination": "goal map, environment truncation, step limit, or battle for navigation-only greedy policies",
        "limitations": ["One fixed initial save and RNG state", "Tile graph discovery uses speculative emulator search",
            "Tile nodes collapse states at identical map coordinates", "Battle handling is outside this navigation test"]}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    for seed in range(episodes):
        for policy in policies:
            print(f"Running {policy}, episode {seed + 1}/{episodes}", flush=True)
            row = run_episode(policy, seed, start_path, rom, model, lookups, graph["edges"], max_steps, output, graph_limit, goal, allowed, cache_dir)
            rows.append(row)
            with (output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(f"{policy}: {row['outcome']}, {row['actual_environment_steps']} actual, {row['simulated_actions']} simulated", flush=True)
    summary = {}
    for policy in policies:
        data = [row for row in rows if row["policy"] == policy]
        summary[policy] = {"episodes": len(data), "successes": sum(row["goal_reached"] for row in data),
            "mean_actual_steps": sum(row["actual_environment_steps"] for row in data) / len(data),
            "mean_simulated_actions": sum(row["simulated_actions"] for row in data) / len(data),
            "mean_wall_seconds": sum(row["wall_seconds"] for row in data) / len(data),
            "mean_unique_coordinates": sum(row["unique_coordinates"] for row in data) / len(data),
            "mean_repeated_states": sum(row["repeated_states"] for row in data) / len(data)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def run(output, rom=DEFAULT_ROM, source=DEFAULT_SOURCE, checkpoint=DEFAULT_CHECKPOINT,
        episodes=3, max_steps=300, graph_limit=1000, policies=DEFAULT_POLICIES,
        cache_dir=None, target="Viridian City", allowed_maps=None, llm_objective=None,
        index_path=ROOT / "pokemon_rag/data/expanded-bulbapedia-clean", start_state=None):
    original_cwd, original_path = Path.cwd(), sys.path[:]
    try:
        return _run(output, rom, source, checkpoint, episodes, max_steps, graph_limit, policies,
                    cache_dir, target, allowed_maps, llm_objective, index_path, start_state)
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rom", type=Path, default=DEFAULT_ROM)
    parser.add_argument("--source-state", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--start-state", type=Path, help="Use this emulator save directly, without Pallet preparation")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--graph-limit", type=int, default=1000)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=DEFAULT_POLICIES)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--target", default="Viridian City")
    parser.add_argument("--allowed-maps", nargs="+")
    parser.add_argument("--llm-objective")
    parser.add_argument("--index-path", type=Path, default=ROOT / "pokemon_rag/data/expanded-bulbapedia-clean")
    args = parser.parse_args()
    run(args.output, args.rom, args.source_state, args.checkpoint, args.episodes, args.max_steps, args.graph_limit, args.policies, args.cache_dir, args.target, args.allowed_maps, args.llm_objective, args.index_path, args.start_state)
