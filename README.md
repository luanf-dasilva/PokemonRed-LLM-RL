# Train RL agents to play Pokemon Red

### New 10-19-24! Updated & Simplified V2 Training Script - See V2 below
### New 1-29-24! - [Multiplayer Live Training Broadcast](https://github.com/pwhiddy/pokerl-map-viz/)  🎦 🔴 [View Here](https://pwhiddy.github.io/pokerl-map-viz/)
Stream your training session to a shared global game map using the [Broadcast Wrapper](/baselines/stream_agent_wrapper.py)  

See how in [Training Broadcast](#training-broadcast) section
  
## Watch the Video on Youtube! 

<p float="left">
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/youtube.jpg?raw=true" height="192">
  </a>
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/poke_map.gif?raw=true" height="192">
  </a>
</p>

## Join the discord server
[![Join the Discord server!](https://invidget.switchblade.xyz/RvadteZk4G)](http://discord.gg/RvadteZk4G)
  
## Running the Pretrained Model Interactively 🎮  
🐍 Python 3.10+ is recommended. Other versions may work but have not been tested.   
You also need to install ffmpeg and have it available in the command line.

### Windows Setup
Refer to this [Windows Setup Guide](windows-setup-guide.md)

### For AMD GPUs
Follow this [guide to install pytorch with ROCm support](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/howto_wsl.html)

### Linux / MacOS

V2 is now recommended over the original version. You may follow all steps below but replace `baselines` with `v2`.

1. Copy your legally obtained Pokemon Red ROM into the base directory. You can find this using google, it should be 1MB. Rename it to `PokemonRed.gb` if it is not already. The sha1 sum should be `ea9bcae617fdf159b045185467ae58b2e4a48b9a`, which you can verify by running `shasum PokemonRed.gb`. 
2. Move into the `baselines/` directory:  
 ```cd baselines```  
3. Install dependencies:  
```pip install -r requirements.txt```  
It may be necessary in some cases to separately install the SDL libraries.
For V2 MacOS users should use ```macos_requirements.txt``` instead of ```requirements.txt```
4. Run:  
```python run_pretrained_interactive.py```
  
Interact with the emulator using the arrow keys and the `a` and `s` keys (A and B buttons).  
You can pause the AI's input during the game by editing `agent_enabled.txt`

Note: the Pokemon.gb file MUST be in the main directory and your current directory MUST be the `baselines/` directory in order for this to work.

## Training the Model 🏋️ 

<img src="/assets/grid.png?raw=true" height="156">


### V2

- Trains faster and with less memory
- Reaches Cerulean
- Streams to map by default
- Other improvements

Replaces the frame KNN with a coordinate based exploration reward, as well as some other tweaks.
1. Previous steps but in the `v2` directory instead of `baselines`
2. Run:
```python baseline_fast_v2.py```

## Tracking Training Progress 📈

### Training Broadcast
Stream your training session to a shared global game map using the [Broadcast Wrapper](/baselines/stream_agent_wrapper.py) on your environment like this:
```python
env = StreamWrapper(
            env, 
            stream_metadata = { # All of this is part is optional
                "user": "super-cool-user", # choose your own username
                "env_id": id, # environment identifier
                "color": "#0033ff", # choose your color :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
```

Hack on the broadcast viewing client or set up your own local stream with this repo:  
  
https://github.com/pwhiddy/pokerl-map-viz/

### Local Metrics
The current state of each game is rendered to images in the session directory.   
You can track the progress in tensorboard by moving into the session directory and running:  
```tensorboard --logdir .```  
You can then navigate to `localhost:6006` in your browser to view metrics.  
To enable wandb integration, change `use_wandb_logging` in the training script to `True`.

## Static Visualization 🐜
Map visualization code can be found in `visualization/` directory.

## Follow up work  
 
Check out our follow up projects & papers!  
  
### [Pokemon Red via Reinforcement Learning 🔗](https://arxiv.org/abs/2502.19920)
```  
  @misc{pleines2025pokemon,
    title={Pokemon Red via Reinforcement Learning},
    author={Marco Pleines and Daniel Addis and David Rubinstein and Frank Zimmer and Mike Preuss and Peter Whidden},
    year={2025},
    eprint={2502.19920},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
  }
```
### [Pokemon RL Edition 🔗](https://drubinstein.github.io/pokerl/)
### [PokeGym 🔗](https://github.com/PufferAI/pokegym)

## Supporting Libraries
Check out these awesome projects!
### [PyBoy](https://github.com/Baekalfen/PyBoy)
<a href="https://github.com/Baekalfen/PyBoy">
  <img src="/assets/pyboy.svg" height="64">
</a>

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
<a href="https://github.com/DLR-RM/stable-baselines3">
  <img src="/assets/sblogo.png" height="64">
</a>

# Walkthrough ingestion

The standalone `pokemon_rag` module extracts prose from the first three parts
of the [Bulbapedia Pokémon Red and Blue walkthrough](https://bulbapedia.bulbagarden.net/wiki/Walkthrough:Pok%C3%A9mon_Red_and_Blue).
Install its current dependencies and run it from the repository root:

```sh
python -m pip install -r pokemon_rag/requirements.txt
python -m pokemon_rag.ingest
```

HTML is cached under `pokemon_rag/data/raw/part_1.html` through `part_3.html`;
the extracted sections are written to `pokemon_rag/data/sections.json`. If the
site denies automated access, save the pages manually at those paths and run the
command again. Tables are excluded, so the output does not include all encounter,
trainer, or item details. The generated data directory is not tracked by Git.

The walkthrough text is contributed by Bulbapedia authors and is subject to
[Bulbapedia's copyright terms](https://bulbapedia.bulbagarden.net/wiki/Bulbapedia:Copyrights),
including its Creative Commons Attribution-NonCommercial-ShareAlike license;
it is not covered by this repository's software license. Each extracted section
retains its source page URL.

## Semantic retrieval

After ingestion, build the MiniLM embeddings and FAISS index, evaluate the
labelled queries, or search for a specific question:

```sh
python -m pokemon_rag.build_index
python -m pokemon_rag.evaluate
python -m pokemon_rag.retriever "What must I do with Oak's Parcel?" -k 4
python -m unittest discover -s tests
```

The first build downloads `sentence-transformers/all-MiniLM-L6-v2`. The model
runs on CPU for indexing and retrieval. The index, chunks, and their integrity
metadata are stored in the ignored `pokemon_rag/data/` directory; rebuild the
index after re-ingesting. The evaluation prints Hit Rate@4 and MRR@4 over the
eight queries in `pokemon_rag/evaluation_cases.json`, then checks whether the
structured-state query retrieves a parcel prerequisite. A nonzero exit status
means the 7/8 hit-rate target or the state-query check was missed. Route 1 is
not labelled relevant for the early-catching question because its extracted
text says Pokémon cannot yet be caught there.

## Structured planner evaluation

With a local Ollama server and `qwen3.5:4b-q4_K_M` available, run the planner
or the paired eight-scenario experiment (five repetitions per condition):

```sh
python -m pokemon_rag.planner
python -m pokemon_rag.evaluate_planner --output pokemon_rag/data/experiments/my-run
python -m pokemon_rag.evaluate_planner --summarize pokemon_rag/data/experiments/my-run
```

The output directory must be new. Each run records the scenario states, model
metadata, retrieved documents, requests, raw responses, timing, and token
counts. Paired conditions have the same prompt and generation settings; only
the walkthrough context differs. Results and local corpus data are Git-ignored.
The index must exist before running the experiment. `--repetitions` and `--seed`
can be set explicitly; their defaults are five and zero.

For a prompt ablation, keep the original run directory and write the revised
run to a new directory. Compare each revised condition with its corresponding
original condition, checking that model metadata, scenarios, generation
settings, and retrieved documents match. The exact prompts are saved in each
trial. These scenarios are development cases, not a held-out test set.
To verify those controls and print the paired summaries, run:

```sh
python -m pokemon_rag.compare_planner_runs path/to/original-run path/to/revised-run
```

`review.jsonl` is a review worksheet keyed by trial ID. A reviewer should use
`protocol.json` for each scenario's expected behavior and `results.jsonl` for
the goal, state, and cited text. Set each applicable label to `true` or `false`,
leave it `null` when it cannot be judged, and explain decisions in `notes`:

- `factual_accuracy`: claims and proposed goal respect known game mechanics.
- `prerequisite_correctness`: the goal does not skip a required earlier event.
- `goal_validity`: the goal is achievable from the supplied state and useful
  toward the objective; an unsupported or overly broad goal can fail here.
- `grounding_accuracy`: in the RAG condition, the cited passages support the
  claims and goal. Leave this `null` without RAG or when it is not assessable.

Run the `--summarize` command after review. Unreviewed labels are excluded from
accuracy rates. Citation-ID checks only test whether IDs were supplied, not
whether they support a claim. The consistency output reports exact agreement of
goal type, normalized target text, and target value; it is not a semantic
equivalence assessment. Completion conditions are descriptive text and are
never evaluated as code.

## Corpus expansion and full-context experiment

Run the original 27 chunks as full context against a saved single-query run:

```sh
python -m pokemon_rag.evaluate_full_context \
  --baseline pokemon_rag/data/experiments/qwen-hierarchical-prompt \
  --output pokemon_rag/data/experiments/full-context
```

The script checks the saved prompt, schema, model digest, and baseline passages,
then probes the actual token count and running context length for every state.
The output records each trial and flags any input-plus-output context overflow.
Review the result worksheet before reporting factual or grounding accuracy.

The separate expanded-corpus workflow reads `pokemon_rag/sources.yaml`. It
includes walkthrough parts 4–6, dedicated articles, and source-linked structured
facts; `--include-later` adds walkthrough parts 7–17. For example:

```sh
python -m pokemon_rag.ingest_expanded --site bulbapedia \
  --output pokemon_rag/data/expanded
python -m pokemon_rag.build_index --data-dir pokemon_rag/data/expanded
python -m pokemon_rag.retriever --data-dir pokemon_rag/data/expanded \
  "Where do I get Oak's Parcel?" -k 4
```

The output must be a new directory, so the original corpus and index remain
available for comparison. The downloader honors robots.txt and stops on access
errors. If a site denies automated access, save its HTML as
`pokemon_rag/data/raw/<site>/<source-id>.html` and use `--offline`; source IDs
are in the catalog. `--site strategywiki` selects the summary and detailed pages
through the third badge. StrategyWiki material is subject to its own licensing
and attribution terms; consult the source site before redistributing it.

General entity articles cover multiple games and generations. They are marked
as mixed scope, and the lightweight filter does not guarantee every extracted
statement applies to Pokémon Red. Tables are still excluded, including the
Generation I type chart. The existing retrieval labels were designed for the
original corpus and need renewed relevance judgments before scoring an expanded
index. The four structured records are non-executable, manually authored facts
with source references, not verified game-state rules.

## Expanded retrieval and state experiments

The following runs expanded-index top 4, top 8, and three-query retrieval, plus
two separate tests using the original full corpus: state placed last and an
inventory/completed-events representation. It verifies the saved prompt,
schema, model digest, and context window before running five repetitions of
each of eight scenarios. The complete result is kept in an ignored directory.

```sh
python -m pokemon_rag.evaluate_variants \
  --baseline pokemon_rag/data/experiments/qwen-hierarchical-prompt \
  --expanded pokemon_rag/data/expanded-bulbapedia-clean \
  --output pokemon_rag/data/experiments/expanded-variants
```

The expanded index contains 274 chunks, but these variants conservatively
retrieve only the 72 Gen I walkthrough or source-grounded curated chunks; they
exclude mixed-generation articles pending passage-level verification. The
multi-query variant interleaves local progression, objective requirements, and
local activities, deduplicating to at most eight chunks. Passing the metadata
filter does not establish that every source claim is factually correct. Review
the worksheet before reporting accuracy or grounding; the automatic
completed-parcel-target count is only a narrow state-contradiction indicator.

## Initial game-state and value-function work

`pokemon_game.pret_data` parses a pinned subset of
[pret/pokered](https://github.com/pret/pokered) into machine-readable lookup
tables, source-linked semantic records, and a directed map graph. Its revision,
selected files, and hashes are in `research/pokered/source_manifest.json`.
To regenerate from a fresh checkout, check out that revision and run:

```sh
python -m pokemon_game.pret_data --source-dir /path/to/pokered
```

The parser verifies the source hashes and writes results to `processed/` under
the source directory. `pokemon_game.state_encoder.encode_state(memory, lookups)`
reads early-game map, position, party, bag, and event fields from a
PyBoy-compatible memory object without mutating it. It separates current bag
contents from recorded event flags. An absent party has unknown HP fraction.

`pokemon_game.value_function.LocationValue` provides a deterministic value for
manually supplied `REACH_LOCATION` goals. It uses graph distance between maps
and Manhattan distance for coordinates on the same map. The graph is partial
and optimistic: it does not model collision, quest locks, or dynamic `LAST_MAP`
warps. It must not be treated as a verified traversable path or used as a
complete greedy policy. Other goal types are explicitly unsupported for now.

## Manual-goal navigation experiment

The emulator-only simulator and directed tile graph can test a manually supplied
Viridian City objective without the LLM. With the local ROM, generated pret
lookups, and a compatible PPO checkpoint available:

```sh
python -m pokemon_game.evaluate_greedy \
  --output pokemon_game/data/navigation-run --episodes 3 --max-steps 300
```

The runner derives one shared Pallet Town snapshot from `has_pokedex.state`;
that save already has a starter and has delivered Oak's Parcel. Source saves and
the ROM are not modified. `--rom`, `--source-state`, `--checkpoint`, and
`--policies` can be provided explicitly. Results, generated saves, and discovered
graphs stay in the ignored `pokemon_game/data/` directory. The protocol records
input hashes, preparation, action sets, and timing.

The simulator uses a second PyBoy instance for speculative actions and never
calls `RedGymEnv.step()` on that instance. The committed run uses the normal
environment step. Tile-graph discovery probes four directions from saved
emulator snapshots and keeps observed directed transitions; the hierarchical
variant combines map-transition distance with normalized distance to an
observed exit. Discovery is multi-step emulator search: its simulated actions
and wall time are included in the planning cost. The model is not a static
collision extraction; it samples one state per map/tile, excludes battles, and
does not establish general traversability under different RNG, NPC, or event
states.

Random and PPO can use all seven environment actions; the navigation-only
greedy policies consider four directions and stop on battle. A 300-step run
from a single emulator snapshot is a development test, not an independent
estimate of success across game states or evidence that greedy lookahead
outperforms a trained policy generally. Repeated policy seeds do not change the
saved emulator RNG state.

## Cached and model-based navigation

Persist a graph and compare cold model-based navigation with warm cached
lookahead and model-based navigation:

```sh
python -m pokemon_game.evaluate_greedy \
  --output pokemon_game/data/cache-run --cache-dir pokemon_game/data/nav-cache \
  --episodes 3 --policies model_greedy cached_greedy
```

The first eligible run discovers and saves a graph; later compatible runs load
it. Compatibility includes the exact initial emulator snapshot, ROM hash, goal
map, allowed maps, action timing, and discovery limit. `model_greedy` follows
known improving edges without speculative lookahead, checks every prediction
against the actual successor, and invalidates or corrects mismatches. It
rediscovers from the current state if the known graph has no path. Battle
states stop navigation. Cache hits, simulated actions, repairs and mismatches
are reported per episode. The sampled graph is still conditioned on emulator
state, and matching coordinates do not prove that other game state matches.

The [cached experiment](research/navigation_experiments/cached-pallet-to-viridian/README.md)
contains a measured cold/warm comparison. For another starting save, use
`--start-state PATH` to load it directly; supply `--target` and
`--allowed-maps` with the map names needed for that route. Graph discovery
stops on the first reached tile of the target map, not a specific indoor tile.

To request a goal from the expanded multi-query RAG planner, run with
`--llm-objective 'Obtain the Boulder Badge'`. This requires the expanded index
at `pokemon_rag/data/expanded-bulbapedia-clean` and a running local Ollama
server. The exact planner response and retrieved documents are saved to
`planner.json`; unsupported goals, unknown maps, invalid evidence and
out-of-scope targets stop the run instead of being replaced by a manual goal.
The included two live planner trials returned `INSUFFICIENT_CONTEXT`; they did
not demonstrate successful LLM-directed navigation.

PPO training optionally accepts a fixed graph potential via a top-level YAML
section (use an absolute path to the graph generated above):

```yaml
navigation_shaping:
  graph: /absolute/path/to/nav-cache/GRAPH_HASH.json
  beta: 1.0
```

The wrapper uses the training `gamma`, zero potential on true termination,
and retains the next-state potential on time-limit truncation. Unknown tiles
have zero potential. The graph is fixed throughout each training run and its
ROM hash must match. This adds shaping to the existing environment reward; it
does not make a dynamic LLM goal policy-invariant or establish any PPO training
improvement. Omit the section for unchanged baseline training.

## Adaptive starting-state curriculum

An independent PPO runner can reset workers from the initial save or from the
best- and worst-return episode snapshots selected during the preceding rollout.
It records the selected states and compares against an otherwise matched
fixed-start control. The implementation, limitations, smoke results and run
commands are described in the [curriculum experiment](research/ppo_experiments/curriculum-smoke/README.md).
The curriculum is separate from the running fixed-goal shaping comparison.
Optional fixed-goal runs exclude already-completed snapshots, and logs record
starting and ending progress as well as transitions by reset source.

## Wild battle prototype

The separate battle environment captures stable command-menu snapshots and
exposes four move actions plus RUN. The [battle experiment record](research/battle_experiments/wild-prototype/README.md)
describes the small archive, controller costs, fixed-policy demonstration,
supported mechanics and a separate PPO runner for later training.

## Demonstrations and imitation

The Gym-based recorder supports human play and model-based expert trajectories.
Offline behavioral cloning and optional imitation updates between PPO rollout
cycles use the same recorded observations and actions. See the [demonstration
experiment record](research/imitation_experiments/greedy-smoke/README.md) for
controls, data format, reproduction commands, smoke results and limitations.
