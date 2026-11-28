# Corrective demonstrations and map-goal conditioning

## Human demonstration

A human played from the post-starter Pallet save to Viridian City through the
Gym recorder. The completed session contains 266 transitions: 84 were labeled
overworld and 182 wild battle. The last transition reached the goal. This is
one demonstration from one starting save, not a population estimate. The local
recording contains the complete observations, actions, rewards, modes and
milestone snapshots; only an aggregate summary is retained here.

A short 10-epoch cloning test combined those 266 actions with the 76-action
greedy demonstration. Offline agreement reached 61.4% on the combined data
and 57.6% on the related Route 1 continuation, but its deterministic policy
still pressed UP for all 300 actions in a Pallet evaluation and did not reach
the goal. This single seed does not establish the effect of human data.

## DAgger-style corrective collection

`pokemon_game.demonstrations.dagger` runs the learner in the real Gym
environment. At each visited state, a separate emulator queries and validates a
model-based expert action; the recorded observation and successor correspond
to the **executed learner or mixed behavior action**, while the supervised
label is the expert action. The two actions and their source are recorded
separately. If the expert cannot produce a supported label within its limits,
it abstains and the episode stops; no substitute label is invented. Labeled
sessions are aggregated with the initial expert sessions and used for a
supervised update after each round. Human, greedy and corrective sources can
also be combined by the periodic imitation trainer.

A known limitation surfaced immediately: a learner action entered Oak's Lab,
and a warp briefly produced coordinates that did not describe a stable tile.
The oracle now tries bounded repeated-direction recovery sequences and verifies
an improving edge on a separate emulator before labeling their first action.
It can also rediscover a bounded local graph. These are approximate labels
conditioned on the sampled emulator state, not guarantees of an optimal or
universally valid action. Every speculative action and frame is recorded.

Two short tests started from the previously collapsed PPO/BC policy:

| Roll-in | Round | Labels | Expert/learner disagreements | Speculative actions | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Learner only | 1 | 8 | 8 | 1,399 | Expert abstained |
| Learner only | 2 | 8 | 8 | 1,399 | Expert abstained |
| 50% expert mixture | 1 | 80 | 70 | 1,142 | 80-step limit |
| 50% expert mixture | 2 | 10 | 10 | 1,446 | Expert abstained |

Both final policies failed to reach Viridian City in a separate 300-action
check. These small runs demonstrate data collection and expose its cost; they
do not show a successful corrective policy or sample-efficiency improvement.
The graph is partial, and teacher discovery overhead is substantial. A learner
can also visit menus or battle states outside the navigation expert's scope.
The oracle stops rather than guessing in those cases.

## Goal-conditioned observations

`GoalConditionedMap` adds a 256-element one-hot map goal to the Gym observation.
It selects an eligible goal at reset, keeps it fixed for the episode, and
terminates on reaching the selected map outside battle. Already-satisfied goals
are excluded at reset. Existing map-labeled demonstrations can be read with
`--goal-conditioned`, which adds the same goal vector to current and successor
observations. Mixed-goal data is only accepted by default when goals are
actually present in the model input. PPO can select map goals with
`--goal-map-ids`; this is map-level goal conditioning, not yet an LLM-driven
training loop or support for non-location objectives.

A 10-epoch smoke test trained a newly initialized goal-conditioned policy on
163 expert transitions across four target maps; its related Route 1 continuation
was held out. It reached 46.0% training action agreement and 61.0% on that
related continuation. In four 300-step live evaluations the model chose UP at
every step and reached none of its goals, including three different goals from
the same Viridian state. Thus the representation works, but this experiment
does not establish that the policy learned to use the goal. Training and
validation trajectories share their origin, and the dataset is too small to
support a generalization claim.

The `*-protocol.json`, `*-rounds.jsonl`, metrics and evaluation summaries
record these development runs. Model files and full demonstrations remain in
ignored local data directories. They are not an independent curriculum or PPO
comparison; those runs are tracked separately.

Example corrective collection (replace paths as needed):

```sh
python -m pokemon_game.demonstrations.dagger \
  --checkpoint PATH_TO_POLICY.zip --start-state PATH_TO_START.state \
  --graph PATH_TO_COMPATIBLE_GRAPH.json --initial-sessions PATH_TO_EXPERT_SESSION \
  --output pokemon_game/data/demonstrations/dagger-run --rounds 2 \
  --expert-probability 0.5
```

For goal-conditioned cloning, pass `--goal-conditioned` to
`pokemon_game.demonstrations.behavioral_cloning` along with map-labeled
sessions. For goal-conditioned PPO, use `--goal-map-ids` instead of
`--goal-map-id` in `pokemon_game.demonstrations.periodic_training`.
