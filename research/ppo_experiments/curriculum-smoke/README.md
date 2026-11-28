# Adaptive reset curriculum

The separate `pokemon_game.curriculum_experiment` runner supports fixed-start
control and adaptive-start PPO without modifying the running fixed-goal
comparison. It defaults to `init.state`, the original game start. Supply a
specific `--start-state` when matching a different experiment.

For 32 adaptive workers, 16 always use the original save, 8 use the latest
best-return final snapshot and 8 use the latest worst-return final snapshot.
Until a valid snapshot is published, those workers use the initial save. At
each rollout boundary, the coordinator picks the highest and lowest **original
environment episode returns among eligible episodes completed in that rollout**.
If no eligible episode completed, the previously published pair remains in
place. A reset already in progress is never interrupted; each worker adopts the
latest pair when it next resets. Exact start fractions can differ temporarily
because episodes end at different times. The fixed-start control uses the same
wrapper and reward handling but assigns all workers to the original save.

Snapshots are saved from the final emulator state after the last action, before
vectorized auto-reset. A single manifest publishes each new pair atomically;
snapshots are checksummed and a missing or corrupted snapshot falls back to the
initial save. Selection checks finite original return, successful save-state
serialization, party size and recognized battle flag. It permits battle states
and does not prove that a state is recoverable or that dialogue and menus are
safe. Battle mode comes from RAM byte `0xD057`: 0 overworld, 1 wild battle,
2 trainer battle, and 255 lost battle, as described by the
[pret/pokered battle constants](https://github.com/pret/pokered/blob/master/constants/battle_constants.asm).
This detection does not implement a separate battle agent.

The base environment recomputes event and reward baselines on every reset. The
wrapper sets previous health and party size from the loaded save, fills the
screen stack with the loaded frame, and allows any outer shaping wrapper to
recompute its potential. Exploration memory and recent actions are reset. The
loaded frame is repeated because the emulator snapshot does not carry the
Python-side observation history; it is not a reconstruction of the previous
three frames. An adaptive start distribution changes the task over time, so
compare policies on the same fixed evaluation saves as well as training data.

## Integration smoke runs

Both runs used four workers, four 32-step rollouts, a 16-step episode limit and
one newly initialized seed-0 policy, for 512 transitions each. The adaptive
run completed 32 episodes: 16 initial, 6 fallback before archive adoption,
5 from published best states and 5 from published worst states. All 32 control
episodes started from the original save. The adaptive run published a new
best/worst pair for each rollout. These very short runs verify mechanics; they
are not a training-performance comparison. Logs and protocols are recorded in
`adaptive/` and `control/`; emulator snapshots and models remain in the ignored
local data directory. After the smoke runs, the mode decoder was extended to
recognize the lost-battle value; that case is covered by a unit test.

Run a matched comparison with the same start, fixed evaluation saves, seeds and
hyperparameters, for example:

```sh
python -m pokemon_game.curriculum_experiment --condition control \
  --output pokemon_game/data/curriculum-control-seed0 --seed 0 \
  --eval-state has_pokedex.state
python -m pokemon_game.curriculum_experiment --condition adaptive \
  --output pokemon_game/data/curriculum-adaptive-seed0 --seed 0 \
  --eval-state has_pokedex.state
```

Defaults are 32 workers, 512 rollout steps, 61 rollouts, and a 300-action
limit; each run therefore uses 999,424 training transitions. Repeat with
matched seeds 1 and 2. `evaluation.jsonl` tracks original return, final map,
position, battle mode and badge count on each fixed save. Full curriculum
comparisons have not been run. They should be scheduled separately from the
ongoing PPO shaping matrix to avoid contending for CPU and confounding wall
time. Ranking different start states by incremental return is itself a
limitation; a low return does not necessarily mean a poor or recoverable state.
