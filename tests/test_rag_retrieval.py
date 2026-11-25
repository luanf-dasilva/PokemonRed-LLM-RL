import contextlib
import hashlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from pokemon_rag.build_index import MAX_TOKENS, MODEL_NAME, OVERLAP, chunk_section, document_text
from pokemon_rag.evaluate import evaluate
from pokemon_rag.retriever import PokemonRetriever


class WordTokenizer:
    def encode(self, text, add_special_tokens=True):
        tokens = re.findall(r"\S+", text)
        return ["[CLS]", *tokens, "[SEP]"] if add_special_tokens else tokens

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class ChunkTests(unittest.TestCase):
    def test_chunks_keep_metadata_limit_and_overlap(self):
        tokenizer = WordTokenizer()
        section = {
            "part": 2, "location": "Viridian City", "section": "Viridian City > Poké Mart",
            "source": "https://example.test/part2", "text": " ".join(f"word{i}" for i in range(500)),
        }
        chunks = chunk_section(section, tokenizer)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(tokenizer.encode(document_text(chunk))), MAX_TOKENS)
            for key in ("part", "location", "section", "source"):
                self.assertEqual(chunk[key], section[key])
        for first, second in zip(chunks, chunks[1:]):
            self.assertEqual(first["text"].split()[-OVERLAP:], second["text"].split()[:OVERLAP])

    def test_too_long_heading_fails(self):
        section = {"section": "heading " * 190, "text": "test"}
        with self.assertRaises(ValueError):
            chunk_section(section, WordTokenizer())


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = Path(self.tmp.name)
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1, 0], [0, 1]], dtype=np.float32))
        faiss.write_index(index, str(self.data_dir / "pokemon.index"))
        self.chunks = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
        self.write_chunks()

    def write_chunks(self):
        (self.data_dir / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")
        manifest = {
            "model": MODEL_NAME,
            "chunks_sha256": hashlib.sha256((self.data_dir / "chunks.json").read_bytes()).hexdigest(),
            "index_sha256": hashlib.sha256((self.data_dir / "pokemon.index").read_bytes()).hexdigest(),
        }
        (self.data_dir / "index_metadata.json").write_text(json.dumps(manifest), encoding="utf-8")

    @patch("pokemon_rag.retriever.SentenceTransformer")
    def test_returns_cosine_scores_and_loads_model_once(self, factory):
        factory.return_value.get_sentence_embedding_dimension.return_value = 2
        factory.return_value.encode.return_value = np.array([[0.8, 0.6]], dtype=np.float32)
        retriever = PokemonRetriever(self.data_dir)
        results = retriever.retrieve("question", k=8)
        self.assertEqual([item["id"] for item in results], ["a", "b"])
        self.assertAlmostEqual(results[0]["score"], 0.8)
        retriever.retrieve("another question", k=1)
        factory.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.assertTrue(factory.return_value.encode.call_args.kwargs["normalize_embeddings"])

    @patch("pokemon_rag.retriever.SentenceTransformer")
    def test_invalid_query_and_k(self, factory):
        factory.return_value.get_sentence_embedding_dimension.return_value = 2
        retriever = PokemonRetriever(self.data_dir)
        for query in ("", "  ", None):
            with self.subTest(query=query), self.assertRaises(ValueError):
                retriever.retrieve(query)
        for k in (0, -1, 1.5, True, None):
            with self.subTest(k=k), self.assertRaises(ValueError):
                retriever.retrieve("question", k=k)

    @patch("pokemon_rag.retriever.SentenceTransformer")
    def test_index_document_mismatch_fails_before_model_load(self, factory):
        self.chunks.pop()
        self.write_chunks()
        with self.assertRaisesRegex(RuntimeError, "count mismatch"):
            PokemonRetriever(self.data_dir)
        factory.assert_not_called()

    @patch("pokemon_rag.retriever.SentenceTransformer")
    def test_model_dimension_mismatch_fails(self, factory):
        factory.return_value.get_sentence_embedding_dimension.return_value = 3
        with self.assertRaisesRegex(RuntimeError, "dimensions mismatch"):
            PokemonRetriever(self.data_dir)

    @patch("pokemon_rag.retriever.SentenceTransformer")
    def test_filter_is_applied_before_limiting_results(self, factory):
        factory.return_value.get_sentence_embedding_dimension.return_value = 2
        factory.return_value.encode.return_value = np.array([[0.8, 0.6]], dtype=np.float32)
        retriever = PokemonRetriever(self.data_dir)
        results = retriever.retrieve("query", k=1, predicate=lambda doc: doc["id"] == "b")
        self.assertEqual([row["id"] for row in results], ["b"])

    @patch("pokemon_rag.retriever.SentenceTransformer")
    def test_changed_documents_fail_even_when_counts_match(self, factory):
        self.chunks.reverse()
        (self.data_dir / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
            PokemonRetriever(self.data_dir)
        factory.assert_not_called()


class EvaluationTests(unittest.TestCase):
    def test_hit_rate_and_reciprocal_rank(self):
        class FakeRetriever:
            def retrieve(self, query, k):
                return [
                    {"id": "a", "score": 0.9, "part": 1, "section": "wrong", "text": ""},
                    {"id": "b", "score": 0.8, "part": 2, "section": "right", "text": "parcel"},
                ]

        cases = [
            {"id": "hit", "query": "q1", "relevant": [{"part": 2, "section": "right", "text_contains": "parcel"}]},
            {"id": "miss", "query": "q2", "relevant": [{"part": 2, "section": "right", "text_contains": "north"}]},
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            hit_rate, mrr = evaluate(FakeRetriever(), cases)
        self.assertEqual(hit_rate, 0.5)
        self.assertEqual(mrr, 0.25)


if __name__ == "__main__":
    unittest.main()
