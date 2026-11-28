# Completed navigation shaping comparison

All 18 prespecified runs completed: six conditions and three matched training
seeds, each with 999,424 environment transitions, 32 workers and a 300-action
episode limit. Each started from the same post-starter Pallet save. The two
fixed evaluation saves are related: the Oak's Lab save was used to construct
the Pallet save. At each checkpoint there is one deterministic evaluation per
save, so these trials are not independent samples from a population of saves.

| Condition | Training successes / episodes | Deterministic successes / evaluations | Mean unknown-potential fraction |
| --- | ---: | ---: | ---: |
| Upstream control | 0 / 9,984 | 0 / 84 | 0% |
| Modified control | 0 / 9,984 | 0 / 84 | 0% |
| Map, beta 1 | 2 / 9,984 | 0 / 84 | 36.3% |
| Tile, beta 0.1 | 0 / 9,984 | 0 / 84 | 59.1% |
| Tile, beta 1 | 0 / 9,984 | 0 / 84 | 58.3% |
| Tile, beta 5 | 13 / 9,984 | 0 / 84 | 49.4% |

No deterministic evaluation succeeded at any checkpoint (0/504). The map
condition's only successful training seed had two successes, first at 825,216
transitions; tile beta 5 had 13 successes in one seed, first at 939,744.
First-success timing is quantized to the 32-worker transition batch. There was
no sustained evaluation success. The two controls had identical episode
sequences and original returns for each matched seed, despite their different
reward switches. This result alone does not explain why those switches had no
observable effect in these runs.

The unknown fraction counts states assigned zero potential because the partial
map or tile model could not score them. High missingness limits how much this
experiment tests useful shaping, and cumulative shaping rewards were negative
in these runs. A true terminal potential of zero was used on entering the goal;
time-limit truncations retain potential for bootstrapping. Shaping was added to
existing base rewards, and wall times include evaluation. A separate exploratory
imitation run and short development checks overlapped part of the matrix, so
wall times are not a clean isolated comparison. The 2,276 speculative actions
and approximately 15.884 seconds used to discover the tile graph were paid
before this experiment and are recorded separately in each tile protocol.

`summary.csv` contains the grouped counts. `runs.json` preserves the per-run
protocols, rollout summaries, evaluation rows, checkpoint hashes and measured
wall times; model weights, ROM and emulator saves remain local ignored files.
Plots are in `plots/`. All curves overlap at zero evaluation success, so the
transition plot does not distinguish the conditions visually. These data do not
show that shaping improved sample efficiency, and they do not rule out an effect
under a better-covered potential, a different task setup, or more diverse saves.
