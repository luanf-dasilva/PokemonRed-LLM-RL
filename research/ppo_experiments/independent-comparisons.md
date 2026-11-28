# Independent curriculum and imitation comparisons

The shaping matrix completed before these comparisons began. A sequential
queue runs two separately controlled experiments across seeds 0, 1 and 2:

1. Fixed-start PPO with imitation disabled versus fixed-start PPO with initial
   and periodic imitation from one greedy and one human demonstration.
2. Fixed-start PPO versus adaptive initial/best/worst resets, both without
   imitation.

Each run uses a newly initialized policy, 32 workers, 512 steps per rollout,
61 rollouts (999,424 environment transitions), batch size 512, one PPO epoch,
gamma 0.997, and a 300-action episode limit. The fixed task is reaching
Viridian City from the post-starter Pallet save. Curriculum snapshots already
on the goal map are excluded. The control and treatment within each pair use
the same runner, observation initialization, start save and hyperparameters;
the curriculum and imitation pairs use different wrappers and should not be
treated as a cross-pair ablation of only one variable.

Each condition is evaluated deterministically from the Pallet training save
and a fixed Oak's Lab save. The latter is related to the training save, so
it is a different starting snapshot rather than an independent game history.
Imitation evaluation also reserves a Route 1 demonstration for action-label
agreement; that trajectory shares an origin with the expert training data.
The runs do not use the goal-conditioned model, DAgger labels, LLM guidance,
battle PPO or potential shaping. Thus the contribution of each intervention
is assessed separately rather than combined with the others.

The queue writes a plan, per-run configurations, logs, `status.json`, models,
training statistics and fixed-state evaluations to an ignored output folder.
Status can be stale if the supervisor is externally terminated. A four-worker,
128-transition-per-condition smoke test completed all four conditions before
the full queue was started; it is an integration test, not evidence of a
learning advantage. As of launch the full results are pending.

Reproduce with:

```sh
python -m pokemon_game.run_learning_comparison \
  --output pokemon_game/data/learning-comparison \
  --start-state pokemon_game/data/cache-benchmark-20261127/pallet-post-starter.state \
  --eval-state has_pokedex.state \
  --expert-session pokemon_game/data/demonstrations/greedy-pallet-viridian-v2 \
  --human-session pokemon_game/data/demonstrations/human-pallet-20261128 \
  --validation-session pokemon_game/data/demonstrations/greedy-route1-v2
```

Those paths are local generated artifacts and require the user-supplied ROM
and recordings. The queue stops on the first failed run; it does not silently
overwrite or resume partially completed experiments.
