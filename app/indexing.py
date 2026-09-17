"""
RepoMind - Phase 1 (suite): embed chunks and store them in ChromaDB.
"""
from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (it's slow to load)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_collection(collection_name: str = "repomind_chunks"):
    """Get or create a ChromaDB collection stored on disk."""
    client = chromadb.PersistentClient(path="chroma_data")
    return client.get_or_create_collection(name=collection_name)


def index_chunks(chunks: list[dict]):
    """Embed a list of code chunks and store them in ChromaDB."""
    if not chunks:
        return

    model = get_model()
    collection = get_collection()

    texts = [c["code"] for c in chunks]
    embeddings = model.encode(texts).tolist()

    ids = [f"{c['file']}::{c['name']}::{c['start_line']}" for c in chunks]
    metadatas = [
        {"file": c["file"], "name": c["name"], "type": c["type"],
         "start_line": c["start_line"], "end_line": c["end_line"]}
        for c in chunks
    ]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Indexed {len(chunks)} chunks.")


if __name__ == "__main__":
    from pathlib import Path
    from chunking import chunk_python_file

    test_file = Path("app/ingestion.py")
    chunks = chunk_python_file(test_file)
    index_chunks(chunks)