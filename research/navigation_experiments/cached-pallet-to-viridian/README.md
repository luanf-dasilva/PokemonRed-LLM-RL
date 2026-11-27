# Cached navigation: Pallet Town to Viridian City

Run on 2026-11-27 using the same prepared post-starter save for all trials. The
first model-based trial starts with an empty cache, and all remaining trials
reuse that graph. The policies alternate in the order shown in results.jsonl;
the seeds do not change the saved emulator RNG state. The cold run's graph is
charged in full to that run, not spread across the warm runs.

| Policy / cache | Success | Actual actions | Simulated actions | Mean wall time |
| --- | --- | ---: | ---: | ---: |
| Model-based / cold (1 run) | 1/1 | 76 | 2,276 | 17.16 s |
| Model-based / warm (2 runs) | 2/2 | 76 | 0 | 1.28 s |
| Cached lookahead / warm (3 runs) | 3/3 | 76 | 304 | 3.98 s |

No model predictions mismatched in these runs. Total action evaluations per
warm model-based run were 76 versus 380 for cached lookahead, excluding the
one-time 2,276-action discovery. Across these three model-based episodes the
amortized total was (2,276 + 3 * 76) / 3 = 834.7 per episode. Wall times
include environment initialization and may vary by machine. Compare the
original separately recorded tile-greedy result (2,656 total evaluations per
run), not a concurrently interleaved baseline. These runs use one game state
and do not establish robustness to different NPCs, story flags, or RNG states.

The raw summary combines cold and warm model-based episodes; do not interpret
its model mean as a warm-cache result. Detailed costs and cache hits are in
results.jsonl. The cache is identified by ROM, exact start snapshot, goal,
allowed maps, action timing and discovery limit; it is not a complete model of
other game states. Predictions are checked after actual actions; a mismatch
invalidates or corrects the edge and can require new discovery.

## Additional routes

The final Viridian City state from a warm forward run was saved and used as the
single common starting state for these cold-cache trials. Each goal and allowed
map set was supplied manually. The Mart and Pokémon Center trials ran
concurrently, so their wall times are not directly comparable to the earlier
serial benchmark. Results and input hashes are in additional-routes.json.

| Route | Success | Actual actions | Discovery actions |
| --- | --- | ---: | ---: |
| Viridian City → Pallet Town | 1/1 | 43 | 3,845 |
| Viridian City → Viridian Mart | 1/1 | 28 | 1,284 |
| Viridian City → Viridian Pokémon Center | 1/1 | 16 | 368 |

These are single trials from one derived save. Forest, Pewter City and Pewter
Gym have not been evaluated. The two recorded RAG planner attempts both
returned `INSUFFICIENT_CONTEXT`, so no LLM-directed success is claimed.
