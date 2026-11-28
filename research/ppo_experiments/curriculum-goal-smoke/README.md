# Curriculum goal and progress accounting smoke test

A two-rollout, four-worker, 256-transition adaptive run used the prepared
Pallet state and map ID 1 (Viridian City) as a fixed goal. The original
shaping experiment was not modified. The new curriculum protocol terminates
all conditions on the fixed goal and excludes final snapshots on that map
from both archive selections. A unit test exercises exclusion directly; this
smoke run did not reach the goal.

Each completed episode records start and end map, badge counts, party levels,
and raw event flag gains/losses, so inherited state is visible separately from
changes during that episode. Raw flags include non-story state and are not a
verified progression score. Per-rollout cumulative transition counts by worker
role and actual snapshot source capture the difference between the configured
50/25/25 worker assignment and the states that actually generated experience.
The second rollout recorded 128 original-state, 16 best-state, 16 worst-state,
and 96 initial-fallback transitions cumulatively. These counts are mechanics
checks, not a performance experiment.
