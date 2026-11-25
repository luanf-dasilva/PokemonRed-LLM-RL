"""Build a CPU MiniLM and FAISS index from extracted walkthrough sections."""

import argparse
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 220
OVERLAP = 30


def document_text(chunk):
    """Include the section hierarchy in both indexing and token counting."""
    return f"{chunk['section']}\n{chunk['text']}"


def chunk_section(section, tokenizer):
    """Split a section into token-bounded text windows with 30-token overlap."""
    text = section["text"]
    prefix = section["section"] + "\n"
    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    budget = MAX_TOKENS - prefix_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if budget <= OVERLAP:
        raise ValueError("Section heading leaves too little room for text and overlap")

    offsets = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )["offset_mapping"]
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + budget, len(offsets))
        while end > start:
            chunk = {**section, "text": text[offsets[start][0]:offsets[end - 1][1]]}
            if len(tokenizer.encode(document_text(chunk), add_special_tokens=True)) <= MAX_TOKENS:
                break
            end -= 1
        if end <= start or (end < len(offsets) and end - start <= OVERLAP):
            raise ValueError("Section heading leaves too little room for text and overlap")
        chunks.append(chunk)
        if end == len(offsets):
            break
        start = end - OVERLAP
    return chunks


def build_index(data_dir=DATA_DIR):
    """Create chunks and normalized embeddings, then write the local index."""
    data_dir = Path(data_dir)
    sections = json.loads((data_dir / "sections.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    chunks = [chunk for section in sections for chunk in chunk_section(section, model.tokenizer)]
    if not chunks:
        raise RuntimeError("No chunks generated")
    for i, chunk in enumerate(chunks):
        chunk["id"] = f"chunk_{i:04d}"

    embeddings = np.ascontiguousarray(model.encode(
        [document_text(chunk) for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    ), dtype=np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    chunks_path = data_dir / "chunks.json"
    index_path = data_dir / "pokemon.index"
    chunks_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    faiss.write_index(index, str(index_path))
    manifest = {
        "model": MODEL_NAME,
        "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    (data_dir / "index_metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding dimensions: {index.d}")
    print(f"Index vectors: {index.ntotal}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    build_index(parser.parse_args().data_dir)
