import json
import tempfile
import unittest
from pathlib import Path

from pokemon_rag.evaluate_variants import historical_state
from pokemon_rag.export_research import export
from pokemon_rag.planner import make_payload
from pokemon_rag.retrieval_experiments import generation_one_verified, retrieve_context


class VariantTests(unittest.TestCase):
    def test_provenance_filter_rejects_mixed_and_unattributed(self):
        good = {"generation": 1, "source": "https://example.test", "site": "bulbapedia",
                "scope": "generation I walkthrough", "game": "Pokémon Red and Blue"}
        self.assertTrue(generation_one_verified(good))
        self.assertFalse(generation_one_verified({**good, "scope": "mixed games and generations"}))
        self.assertFalse(generation_one_verified({**good, "source": ""}))
        self.assertFalse(generation_one_verified({**good, "game": "Pokémon Red, Blue, and Yellow"}))

    def test_multi_query_coverage_and_deduplication(self):
        class Retriever:
            def retrieve(self, query, k, predicate):
                self_test.assertEqual(k, 4)
                self_test.assertTrue(callable(predicate))
                if "requirements" in query:
                    return [{"id": "objective"}, {"id": "shared"}, {"id": "d"}, {"id": "e"}]
                if "activities" in query:
                    return [{"id": "local"}, {"id": "activity"}, {"id": "f"}, {"id": "g"}]
                return [{"id": "local"}, {"id": "shared"}, {"id": "a"}, {"id": "b"}]
        self_test = self
        result = retrieve_context(Retriever(), {"location": "Viridian City"}, "Boulder Badge", "multi")
        ids = [row["id"] for row in result]
        self.assertEqual(ids[:3], ["local", "objective", "shared"])
        self.assertIn("activity", ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(len(ids), 8)

    def test_history_keeps_inventory_and_completed_events_distinct(self):
        state = {"location": "Viridian Forest", "has_starter": True,
                 "has_oaks_parcel": False, "delivered_oaks_parcel": True}
        converted = historical_state(state)
        self.assertEqual(converted["inventory"]["oaks_parcel"], 0)
        self.assertIn("Obtained Oak's Parcel", converted["completed_events"])
        self.assertIn("Delivered Oak's Parcel", converted["completed_events"])
        before = historical_state({**state, "has_oaks_parcel": False, "delivered_oaks_parcel": False})
        self.assertNotIn("Obtained Oak's Parcel", before["completed_events"])

    def test_placement_changes_order_but_not_context(self):
        state = {"location": "Viridian City"}
        docs = [{"id": "c", "section": "Route", "text": "Go north."}]
        first = make_payload(state, "Badge", docs, placement="state_first")
        last = make_payload(state, "Badge", docs, placement="state_last")
        self.assertTrue(first["messages"][1]["content"].startswith("CURRENT STATE:"))
        self.assertTrue(last["messages"][1]["content"].startswith("WALKTHROUGH CONTEXT:"))
        self.assertEqual(first["format"], last["format"])
        self.assertIn("[c]\nSection: Route\nGo north.", last["messages"][1]["content"])
        self.assertEqual(first["messages"][0], last["messages"][0])

    def test_research_export_omits_source_passages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"trial_id": "P01", "goal": {"target": "Lab"},
                   "documents": [{"id": "chunk_1", "text": "source passage"}]}
            (root / "results.jsonl").write_text(json.dumps(row) + "\n")
            target = root / "export.jsonl"
            self.assertEqual(export(root, target), 1)
            exported = json.loads(target.read_text())
            self.assertEqual(exported["context_ids"], ["chunk_1"])
            self.assertNotIn("source passage", target.read_text())


if __name__ == "__main__":
    unittest.main()
