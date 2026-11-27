# Pallet Town to Viridian City: development test

The two recorded runs share the same ROM, checkpoint, source save, derived
Pallet Town save, action timing, and 300-step limit, verified by their hashes.
The source save already has a starter and the parcel has been delivered. The
goal is to enter Viridian City; this is not a first-time progression evaluation.

The flat tile policy uses directed shortest paths in a graph discovered by
speculatively exploring emulator snapshots. The hierarchical policy combines
map-transition distance with normalized tile distance to an observed exit.
Both count graph discovery and per-action lookahead as simulated actions. The
tile graph is not pre-supplied for free. This search uses future emulator
transitions from the test's fixed start state and is not evidence of a
generalizable static collision map.

Random and PPO use seven actions, while the greedy policies consider four
directions and stop if a battle begins. PPO uses one existing checkpoint on CPU
with deterministic predictions; it was not retrained for this test. All three
episodes per policy start from the same emulator state and RNG state. Only the
random policy's action-sampling seed varies, so the repeated deterministic
episodes are not independent samples of game situations.

`summary.json` and `outcomes.jsonl` contain the metrics and action sequences;
`protocol.json` records input hashes and configuration. ROM, save, checkpoint,
and discovered tile-graph files are not tracked. Trial wall time includes the
per-episode environment setup, graph discovery, and execution; common source
save preparation and one-time PPO checkpoint loading are recorded separately.
