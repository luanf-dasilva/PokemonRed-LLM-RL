# Planner experiments

These files preserve the configurations, scenario definitions, and automated
summaries for the initial prompt, hierarchical prompt, and full-context runs.
Full trial records and the downloaded corpus remain in the ignored
`pokemon_rag/data/` directory. The local full-context results contain 40 trials.
The full-context `outcomes.jsonl` preserves all 40 generated goals, citation IDs,
and measured statistics without copying the walkthrough passages. Reviewing
citation support still requires the corresponding source corpus.

Factual accuracy, prerequisite correctness, goal validity, and grounding
accuracy were not independently reviewed; their summary values are null.
Citation ID validity checks only whether IDs were supplied to the model.
Latency is local inference time and excludes context retrieval. These eight
scenarios are development cases and are not a held-out evaluation set.

`qwen-expanded-variants` contains 200 sanitized outcomes across five modes.
Its expanded-index modes use 72 provenance-filtered chunks from the 274-chunk
index, while the placement and history modes use the original 27 chunks. The
`completed_parcel_target` statistic is a narrow literal-target check, not a
complete state-contradiction or accuracy metric. Summaries contain no human
accuracy or grounding labels until the ignored review worksheet is completed.
