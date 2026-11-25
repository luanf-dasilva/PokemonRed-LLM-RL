import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import ValidationError

from pokemon_rag.evaluate_planner import summarize
from pokemon_rag.compare_planner_runs import compare
from pokemon_rag.planner import PokemonPlanner, make_payload
from pokemon_rag.schemas import Goal


STATE = {"location": "Viridian City", "has_oaks_parcel": False}
DOCS = [{"id": "chunk_1", "section": "Viridian City > Poké Mart", "text": "Collect the parcel."}]
GOAL = {"goal_type": "REACH_LOCATION", "target": "Poké Mart", "target_value": None,
        "reason": "Collect the parcel.", "completion_condition": "Enter the Poké Mart.", "evidence": ["chunk_1"]}


class SchemaTests(unittest.TestCase):
    def test_schema_rejects_unknown_goal_and_extra_fields(self):
        for update in ({"goal_type": "TELEPORT"}, {"unwanted": True}, {"target_value": "12"}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                Goal.model_validate({**GOAL, **update})

    def test_optional_numeric_target(self):
        self.assertEqual(Goal.model_validate({**GOAL, "target_value": 12}).target_value, 12)
        self.assertIsNone(Goal.model_validate(GOAL).target_value)


class PlannerTests(unittest.TestCase):
    def test_pair_changes_only_context(self):
        without = make_payload(STATE, "Boulder Badge", [])
        with_rag = make_payload(STATE, "Boulder Badge", DOCS)
        self.assertEqual(without.keys(), with_rag.keys())
        for key in without.keys() - {"messages"}:
            self.assertEqual(without[key], with_rag[key])
        self.assertEqual(without["messages"][0], with_rag["messages"][0])
        self.assertEqual(with_rag["messages"][1]["content"],
                         without["messages"][1]["content"] + "[chunk_1]\nSection: Viridian City > Poké Mart\nCollect the parcel.")

    @patch("pokemon_rag.planner.requests.post")
    def test_structured_response_and_unknown_citation(self, post):
        response = Mock()
        response.json.return_value = {"message": {"content": json.dumps(GOAL)}, "eval_count": 20}
        post.return_value = response
        retriever = Mock()
        retriever.retrieve.return_value = DOCS
        planner = PokemonPlanner(retriever=retriever)
        self.assertEqual(planner.plan(STATE, "Boulder Badge").target, "Poké Mart")
        self.assertTrue(post.call_args.kwargs["json"]["format"]["properties"])
        response.json.return_value["message"]["content"] = json.dumps({**GOAL, "evidence": ["invented"]})
        with self.assertRaisesRegex(ValueError, "not supplied"):
            planner.plan(STATE, "Boulder Badge")

    @patch("pokemon_rag.planner.requests.post")
    def test_invalid_schema_is_retained_for_evaluation(self, post):
        response = Mock()
        response.json.return_value = {"message": {"content": "not JSON"}}
        post.return_value = response
        result = PokemonPlanner().generate(STATE, "Boulder Badge", [])
        self.assertFalse(result["schema_valid"])
        self.assertEqual(result["raw_content"], "not JSON")
        self.assertIsNone(result["goal"])

    @patch("pokemon_rag.planner.requests.post")
    def test_unconditioned_plan_does_not_load_retriever(self, post):
        response = Mock()
        response.json.return_value = {"message": {"content": json.dumps({**GOAL, "evidence": []})}}
        post.return_value = response
        planner = PokemonPlanner()
        self.assertEqual(planner.plan(STATE, "Boulder Badge", use_rag=False).evidence, [])
        self.assertIsNone(planner.retriever)


class SummaryTests(unittest.TestCase):
    def test_unknown_human_reviews_are_not_counted_as_passes(self):
        rows = [
            {"trial_id": "1", "condition": "with_rag", "scenario_id": "P01", "schema_valid": True,
             "goal": GOAL, "prompt_eval_count": 200, "eval_count": 40, "latency_seconds": 1},
            {"trial_id": "2", "condition": "with_rag", "scenario_id": "P01", "schema_valid": True,
             "goal": GOAL, "prompt_eval_count": 300, "eval_count": 60, "latency_seconds": 3},
        ]
        reviews = [{"trial_id": "1", "factual_accuracy": True}, {"trial_id": "2", "factual_accuracy": None}]
        stats = summarize(rows, reviews)["with_rag"]
        self.assertEqual(stats["schema_valid_rate"], 1)
        self.assertEqual(stats["factual_accuracy"], {"reviewed": 1, "pass_rate": 1.0})
        self.assertEqual(stats["grounding_accuracy"], {"reviewed": 0, "pass_rate": None})
        self.assertEqual(stats["mean_prompt_tokens"], 250)
        self.assertTrue(stats["exact_goal_consistency"]["P01"])


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old, self.new = (Path(self.tmp.name) / name for name in ("old", "new"))
        self.protocol = {"model": "test", "model_info": {"digest": "abc"}, "objective": "goal",
                         "repetitions": 1, "seed": 0, "case_file_sha256": "hash", "cases": [{"id": "P01"}]}
        for path in (self.old, self.new):
            path.mkdir()
            (path / "protocol.json").write_text(json.dumps(self.protocol))
            self.write_results(path, "old prompt" if path == self.old else "new prompt")

    def write_results(self, path, prompt, documents=None):
        records = []
        for condition in ("without_rag", "with_rag"):
            records.append({
                "trial_id": f"P01-{condition}-1", "condition": condition, "scenario_id": "P01",
                "documents": (documents or []) if condition == "with_rag" else [],
                "request": {"model": "test", "messages": [{"role": "system", "content": prompt},
                                                                {"role": "user", "content": "state"}]},
                "goal": GOAL, "schema_valid": True,
            })
        (path / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))

    def test_matches_prompt_only_runs(self):
        report = compare(self.old, self.new)
        self.assertEqual(report["revised"]["with_rag"]["trials"], 1)

    def test_rejects_changed_context_and_duplicate_trials(self):
        self.write_results(self.new, "new prompt", documents=DOCS)
        with self.assertRaisesRegex(ValueError, "Retrieved documents differ"):
            compare(self.old, self.new)
        self.write_results(self.new, "new prompt")
        path = self.new / "results.jsonl"
        lines = path.read_text().splitlines()
        path.write_text(lines[0] + "\n" + lines[0] + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate trials"):
            compare(self.old, self.new)

    def test_rejects_changed_model_and_unchanged_prompt(self):
        protocol = {**self.protocol, "model_info": {"digest": "different"}}
        (self.new / "protocol.json").write_text(json.dumps(protocol))
        with self.assertRaisesRegex(ValueError, "Model digests"):
            compare(self.old, self.new)
        (self.new / "protocol.json").write_text(json.dumps(self.protocol))
        self.write_results(self.new, "old prompt")
        with self.assertRaisesRegex(ValueError, "distinct system prompt"):
            compare(self.old, self.new)


if __name__ == "__main__":
    unittest.main()
