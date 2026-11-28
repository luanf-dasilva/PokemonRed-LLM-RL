# Wild battle prototype

This development test captures Pokémon Red wild encounters at the on-screen
FIGHT/RUN command menu, rather than at the earlier RAM battle flag transition.
The controller exposes four move slots and RUN. It checks move presence, PP,
and the disabled-move index, penalizes an invalid move without advancing the
emulator, and bounds dialogue/menu processing. Each returned transition logs
abstract actions, Game Boy controller inputs and emulator frames. Outcomes are
victory, defeat, escape, unknown exit, controller error, or timeout. The
30-element observation contains active player/opponent HP fractions, levels,
status, species and types, the player's moves and PP, combat stats, disabled
move and episode budget. The simple experimental reward combines normalized HP
changes with +1 for a victory and -1 for a defeat; this reward has not been
validated as a learning objective.

Scope is deliberately narrow: English Gen I menu/tile layout, ordinary wild
battles, one Pokémon in the party, no inventory, capture, party switching,
trainer encounters, move learning or automatic Struggle. Held inputs during
text originally caused extra moves in a prototype; the current controller
releases text input immediately and the tests verify the requested move's PP
changes by exactly one. Unexpected menu transitions are bounded and reported
as truncations rather than silently attributed to the policy. The environment
has no claim of reliability across all battle mechanics. RAM structure and
battle menu logic were checked against [pret/pokered's RAM macros](https://github.com/pret/pokered/blob/master/macros/ram.asm)
and [battle engine](https://github.com/pret/pokered/blob/master/engine/battle/core.asm).

## Archive and demonstration

A fixed exploration action trace was replayed from the post-starter Pallet save.
Varying a short wait before action 65 produced distinct encounter/RNG states.
The training archive has five command-menu states: three successive decisions
in one Pidgey encounter, one Rattata encounter and one Pidgey encounter. The
held-out archive has two other captured encounters, one Rattata and one Pidgey.
All share the same original save and exploration trace, and the successive
Pidgey states share one encounter; the split is not independent player or game
progress. Manifest files here preserve metadata and hashes; the actual emulator
snapshots are local, ignored artifacts under `pokemon_game/data/` and are not
included in the repository. The protocol records the capture cost separately
from battle control. Replaying a trace with a different wait need not encounter
a battle at all.

| Policy | Archive | Outcomes | Mean abstract actions | Mean emulator frames |
| --- | --- | --- | ---: | ---: |
| First available move | Train | 5 victories / 5 | 2.4 | 1,035.2 |
| First available move | Held-out | 2 victories / 2 | 3.5 | 1,484.0 |
| Random legal action | Held-out, 3 repeats per state | 1 victory, 5 escapes | 3.33 | 1,176.0 |

These are hand-written policies, not trained agents. The random policy includes
RUN, so escape is not a victory. Each repetition starts from the same saved
emulator RNG state. This test demonstrates the controller interface, not RL
sample efficiency or general battle competence. Raw outcomes and protocols are
in the adjacent directories.

Recreate captures and archive splits with:

```sh
python -m pokemon_game.capture_battles \
  --start-state pokemon_game/data/cache-benchmark-20261127/pallet-post-starter.state \
  --trace research/navigation_experiments/pallet-to-viridian/outcomes.jsonl \
  --output pokemon_game/data/battle-v2-0 --derive-moves 2
# Repeat with --wait-before-action 65 and --wait-frames 6, 8, 11 and 14,
# using a new output path for each capture. Use 0, 6 and 8 for training;
# use 11 and 14 for held-out evaluation.
python -m pokemon_game.battle_archive --output pokemon_game/data/battle-train \
  pokemon_game/data/battle-v2-0 pokemon_game/data/battle-v2-6 pokemon_game/data/battle-v2-8
python -m pokemon_game.battle_archive --output pokemon_game/data/battle-heldout \
  pokemon_game/data/battle-v2-11 pokemon_game/data/battle-v2-14
python -m pokemon_game.evaluate_battles --archive pokemon_game/data/battle-heldout \
  --output pokemon_game/data/battle-evaluation --policy first_move
```

A separate, untrained PPO runner is provided for a later experiment:

```sh
python -m pokemon_game.train_battle_ppo --rom PokemonRed.gb \
  --train-archive pokemon_game/data/battle-train --eval-archive pokemon_game/data/battle-heldout \
  --output pokemon_game/data/battle-ppo-seed0 --seed 0
```

It uses an MLP and logs abstract decisions, controller inputs, emulator frames,
invalid moves and held-out outcomes. Standard PPO samples the unmasked action
space; invalid moves incur a penalty. Its discount factor applies per abstract
battle decision, not per emulator frame. No battle PPO training was launched
while the navigation-shaping matrix was running.
