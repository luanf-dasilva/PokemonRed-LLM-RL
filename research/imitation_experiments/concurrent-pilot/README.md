# Concurrent PPO and imitation pilot

On request, this exploratory one-worker PPO run was executed concurrently
with the 32-worker navigation shaping matrix, at reduced process priority.
It used the same Pallet-to-Viridian starting save and one 76-transition greedy
expert session, with 10 initial cloning epochs, 20 PPO rollouts of 512 steps,
and three imitation epochs after the first ten rollouts. PPO used batch size
512, one epoch per rollout and gamma 0.997. The run completed 10,240 training
transitions in 37.5 seconds; its wall time and the simultaneously running
matrix's wall time are affected by contention and are not directly comparable
to isolated runs. No human demonstration, adaptive resets, battle policy,
LLM guidance, or reward shaping was used in this pilot.

The training demonstration contains 49 upward actions out of 76, so a policy
that always presses up attains 64.5% offline action agreement. The related
Route 1 continuation contains 36 upward actions out of 59 (61.0%). The cloned
policy achieved exactly these agreement rates and did not improve them at the
periodic update. At the initial checkpoint, after the periodic update, and at
the final checkpoint, deterministic evaluations from both the Pallet and
Route 1 starts chose up for all 300 actions and reached neither goal. The
Route 1 state is derived from the training trajectory, not an independent
held-out game state. There is one seed and no matched no-imitation run, so the
pilot cannot establish the effect of imitation on learning efficiency.

The small protocol, schedule and evaluation summaries are included here;
checkpoints and full observations remain in the ignored local
`pokemon_game/data/demonstrations/` directory. This is evidence of a degenerate
majority-action policy in this pilot, not evidence that the complete research
architecture has been evaluated.
