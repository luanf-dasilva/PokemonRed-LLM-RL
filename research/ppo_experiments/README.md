# Fixed-goal PPO study

`pokemon_game.ppo_navigation_experiment` runs a single newly initialized PPO
policy. `pokemon_game.run_ppo_matrix` queues the following conditions for each
of seeds 0, 1 and 2, in the order shown:

| Condition | Base reward configuration | Potential | Beta |
| --- | --- | --- | ---: |
| Modified control | `configs/baseline.yaml` reward switches | None | 0 |
| Tile-guided | Modified | Directed tile distances | 1.0 |
| Upstream control | `configs/upstream.yaml` reward switches | None | 0 |
| Map-guided | Modified | Partial map graph distance | 1.0 |
| Tile-guided weak | Modified | Directed tile distances | 0.1 |
| Tile-guided strong | Modified | Directed tile distances | 5.0 |

The base environment's other settings are held fixed, including reward scale,
exploration weight and seven actions. Each run uses 32 workers, 512 steps per
rollout, 61 complete rollouts (999,424 transitions), batch size 512, one PPO
epoch, gamma 0.997, and an identical 300-step episode limit. Entering Viridian
City outside battle ends an episode in every condition; the true terminal
potential is zero. Time limits are truncations and retain the observed next
potential for bootstrapping. LLM goals are not used. Map and tile potentials
are fixed during training, initialized once in the parent process, and copied
to the workers. Unknown potential states are zero and explicitly counted.

Deterministic evaluation occurs before training, after every five completed PPO
updates, and after the final update. There is one episode per evaluation save;
the original Oak's Lab save is a different starting snapshot, but it is related
to the training snapshot and is not an independent RNG or population sample.
Aggregate outcomes across seeds as fractions of seeds succeeding at each
checkpoint, with the limited save coverage stated. For a reproducible exploratory
sustained-success metric, require success at three consecutive evaluation
checkpoints on the training save; report the first checkpoint of that sequence.
Report first successful training episode separately; its timestep is resolved
only to the vectorized 32-transition batch. If success is never observed, report
it as not reached within budget rather than as zero time.

Each run records input and code hashes, `protocol.json`, cumulative original
and shaping rewards and missing-state counts in `training.jsonl`, individual
completed training episodes in `episodes.jsonl`, deterministic evaluations in
`evaluation.jsonl`, and a final model and `complete.json`. Evaluation wall
time is included in elapsed wall time. The tile graph was built beforehand;
its observed 15.884-second, 2,276-simulated-action discovery cost should be
reported separately or added explicitly to the tile-guided wall-time axis.
It is not automatically charged once to each episode or independently rebuilt
by each worker. The modified control is the direct comparison for tile shaping.

The [short integration pilots](navigation-pilot/README.md) are not part of the
million-transition comparison. The full matrix can be launched with:

```sh
python -m pokemon_game.run_ppo_matrix \
  --output pokemon_game/data/ppo-navigation-matrix \
  --start-state pokemon_game/data/cache-benchmark-20261127/pallet-post-starter.state \
  --graph pokemon_game/data/cache-benchmark-20261127-cache/GRAPH_HASH.json \
  --eval-state has_pokedex.state --graph-discovery-seconds 15.884149915
```

Use the actual cached graph filename. The parent process writes `plan.json` and
`matrix-status.json`, plus one log and output directory per condition and seed;
it stops on the first failed run. The output path must not already exist.
A complete comparison requires all 18 runs; individual pilots and partial
runs do not answer the sample-efficiency question.

Generate a CSV and plots from completed runs with:

```sh
python -m pokemon_game.plot_ppo_results pokemon_game/data/ppo-navigation-matrix \
  --output pokemon_game/data/ppo-navigation-plots --include-graph-cost
```

Pass `--save unseen_1` for the Oak's Lab evaluation, or
`--include-incomplete` to inspect in-progress data. The transition plot shows
the fraction of available seeds that succeed at each checkpoint; the CSV
includes the number of contributing seeds. The wall-time plot shows each seed
separately. A status file can be stale after an external process termination;
check for a live process and advancing per-run logs before treating it as active.
