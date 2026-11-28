# Demonstration recording and imitation smoke tests

The recorder wraps the existing seven-action `RedGymEnv`; actions always go
through its `step()` and observations use its existing preprocessing. Each
transition is an atomically written, checksummed NumPy archive containing exact
current and next dictionary observations, action, original reward, reward
component deltas, map/coordinates, game mode, party, inventory, known event
flags, source, and termination/truncation flags. A manifest records provenance
and milestones. Emulator snapshots are saved at episode start/end, map/mode or
known-event changes, and on manual request. Closing an unfinished human session
is recorded as a session end rather than inventing an environment termination.
Local demonstration files and emulator saves are ignored under
`pokemon_game/data/`; this directory contains manifests and small results, not
the original game states or full observations.

Start a human recording from the original state:

```sh
python -m pokemon_game.demonstrations.human_player \
  --start-state init.state --output pokemon_game/data/demonstrations/human-001
```

Use arrow keys, Z for A, X for B, and Enter for Start. S saves a snapshot, R
resets after an episode ends, and Q finishes the session. The window shows
location, coordinates, mode, party health/levels, badges, reward, and recorded
transition count. To start from a curriculum's current worst snapshot, replace
`--start-state` with `--curriculum-archive PATH_TO_ARCHIVE`. The snapshot's
hash is verified. `--target-map 1` makes Viridian City a terminal goal and
labels the session. A brief automated GUI test entered two actions and exited;
it was a scripted UI test, not a human demonstration.

Five model-based expert sessions were recorded through Gym without online
speculation. Four reproduce the previously tested routes (76, 43, 28 and 16
actions); another starts at a Route 1 snapshot from the first trajectory and
reaches Viridian City in 59 actions. Every executed graph prediction matched.
The Route 1 run explicitly opts into a different starting snapshot and checks
each actual transition; sharing a coordinate does not prove the original
cached transition model applies to all game states. Exact initial save hashes,
goals, action counts, checksums and milestone metadata are in `sessions/`.

## Behavioral cloning smoke test

A newly initialized PPO `MultiInputPolicy` received 10 supervised epochs on the
76-action Pallet-to-Viridian session. A separate Route 1 session, starting from
a saved point on the *same trajectory*, was reserved for validation. Offline
deterministic action agreement increased from 13.2% to 64.5% on training and
from 15.3% to 61.0% on that related validation session. These are action
classification rates, not episode success rates. The validation trajectory
shares its origin and route with training and is not an independent sample.

When evaluated without the graph, the cloned policy reached neither goal in
300 Gym steps: from Pallet it ended in Oak's Lab, and from Route 1 it remained
on Route 1. Thus this short test did **not** distill a successful navigation
policy. See `bc-metrics.json` and the two evaluation JSON files. Five recorded
sessions have different goal maps, and the existing PPO observations do not
encode goals. The loader rejects mixed goal labels, including a mix of labeled
and unlabeled sessions, by default; an
explicit `--allow-mixed-goals` override is only for deliberate exploratory
experiments, not a substitute for goal-conditioned observations.

## Periodic imitation

`configs/imitation_example.yaml` supplies the requested 10 initial epochs,
three periodic epochs every ten PPO rollouts, batch size 512 and learning rate
0.0001. Run a separate experiment, for example:

```sh
python -m pokemon_game.demonstrations.periodic_training \
  --config configs/imitation_example.yaml --start-state PATH_TO_START.state \
  --goal-map-id 1 --sessions PATH_TO_GREEDY_OR_HUMAN_SESSION \
  --demonstration-root PATH_TO_FINALIZED_SESSION_DIRECTORY \
  --output pokemon_game/data/demonstrations/periodic-run
```

Only enabled sources are loaded; successful PPO data must be explicitly
marked successful, and failed or interrupted expert sessions are rejected.
The optional root is rescanned at each imitation update, so a newly finalized
human session can be added while a separate training run is active. Do not put
validation sessions or unrelated goals in that root. Train and validation
sessions must have different exact initial snapshot hashes, though that alone
does not prove independence.

Each imitation phase uses its own Adam optimizer and runs only after a completed
PPO `learn()` chunk, before the next fresh on-policy rollout. PPO and imitation
checkpoints and `schedule.jsonl` are written at each boundary. The example
uses a constant PPO learning rate; calling `learn()` in chunks changes how a
nonconstant PPO schedule would be interpreted, so this runner checks checkpoint
hyperparameters and should not be used with such a schedule. A one-worker,
48-transition test performed initial cloning, PPO, imitation, PPO, imitation,
then PPO; `periodic-schedule.jsonl` records the order. It does not establish a
training improvement. No long imitation run was started alongside the ongoing
shaping matrix.
