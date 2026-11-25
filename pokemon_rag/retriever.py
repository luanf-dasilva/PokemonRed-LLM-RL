"""Retrieve walkthrough chunks by cosine similarity on the CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pokemon_rag.build_index import DATA_DIR, MODEL_NAME


class PokemonRetriever:
    def __init__(self, data_dir=DATA_DIR):
        data_dir = Path(data_dir)
        index_path = data_dir / "pokemon.index"
        chunks_path = data_dir / "chunks.json"
        manifest = json.loads((data_dir / "index_metadata.json").read_text(encoding="utf-8"))
        if (
            manifest.get("model") != MODEL_NAME
            or manifest.get("chunks_sha256") != hashlib.sha256(chunks_path.read_bytes()).hexdigest()
            or manifest.get("index_sha256") != hashlib.sha256(index_path.read_bytes()).hexdigest()
        ):
            raise RuntimeError("Index and document metadata mismatch; rebuild the index")
        self.index = faiss.read_index(str(index_path))
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        if self.index.ntotal != len(self.chunks):
            raise RuntimeError("Index and document count mismatch; rebuild the index")
        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty; rebuild the index")
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise RuntimeError("Index must use inner product; rebuild the index")
        ids = [chunk.get("id") for chunk in self.chunks]
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in ids) or len(set(ids)) != len(ids):
            raise RuntimeError("Chunk IDs are missing or duplicated; rebuild the index")
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        if self.index.d != self.model.get_sentence_embedding_dimension():
            raise RuntimeError("Index and model dimensions mismatch; rebuild the index")

    def retrieve(self, query, k=4):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a nonempty string")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        query_vector = np.ascontiguousarray(self.model.encode(
            [query], normalize_embeddings=True
        ), dtype=np.float32)
        scores, indices = self.index.search(query_vector, min(k, len(self.chunks)))
        return [
            {**self.chunks[int(index)], "score": float(score)}
            for score, index in zip(scores[0], indices[0])
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question or game-state search query")
    parser.add_argument("-k", type=int, default=4, help="Number of results")
    args = parser.parse_args()
    print(json.dumps(PokemonRetriever().retrieve(args.query, args.k), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
