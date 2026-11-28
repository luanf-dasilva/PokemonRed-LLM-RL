# PPO navigation integration pilots

Two newly initialized seed-0 PPO policies ran one 16,384-transition rollout
(32 workers, 512 steps each, batch size 512, one PPO epoch, gamma 0.997) from
the same post-starter save. Both use the modified base rewards; only the tile
run adds beta=1 potential shaping. Both terminate on entering Viridian City
and truncate after 300 steps. The rollout is gathered before the first PPO
update, so its matched action trajectory is expected to be identical and does
not measure a learning advantage.

| Measure | Modified control | Tile beta=1 |
| --- | ---: | ---: |
| Training episodes completed | 32 | 32 |
| Training successes | 0 | 0 |
| Cumulative original reward | 19.1625 | 19.1625 |
| Cumulative shaping reward | 0 | -0.7300 |
| Unknown-potential successor states | 0 | 7,335 / 16,384 |
| Wall time, including initialization and evaluation | 28.11 s | 28.78 s |

The deterministic policy failed to reach the goal in one evaluation on the
training save after the update in each run. The tile run also evaluated the
original Oak's Lab save, a different starting snapshot from the derived
training save, and did not reach the goal. These are single deterministic
episodes, not independent success-rate estimates. Source/protocol records and
raw logs are in the respective subdirectories. The original graph was built
in a separate navigation experiment, not during these pilots; its one-time
discovery cost was 2,276 simulated actions and approximately 15.88 seconds.

Missing graph coordinates use zero potential. The high fallback frequency is
material: the discovered graph is partial and the pilot does not establish
that tile shaping conveys useful progress information to PPO. The two pilots
used different sets of evaluation saves, so their wall times are not a paired
training-speed comparison. Full runs should use the same evaluation save set.
