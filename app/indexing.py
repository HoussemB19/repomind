"""
RepoMind - Phase 1 (suite): embed chunks and store them in ChromaDB.
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_collection(collection_name: str = "repomind_chunks"):
    """Get or create a ChromaDB collection stored on disk."""
    project_root = Path(__file__).resolve().parent.parent
    chroma_path = Path(os.getenv("CHROMA_DATA_PATH", "chroma_data"))
    if not chroma_path.is_absolute():
        chroma_path = project_root / chroma_path
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(name=collection_name)


def index_chunks(chunks: list[dict], repo_url: str = "unknown"):
    """Embed a list of code chunks and store them in ChromaDB."""
    if not chunks:
        return

    model = get_model()
    collection = get_collection()

    texts = [c["code"] for c in chunks]
    embeddings = model.encode(texts).tolist()

    ids = [f"{repo_url}::{c['file']}::{c['name']}::{c['start_line']}" for c in chunks]
    metadatas = [
        {"file": c["file"], "name": c["name"], "type": c["type"],
         "start_line": c["start_line"], "end_line": c["end_line"], "repo": repo_url}
        for c in chunks
    ]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Indexed {len(chunks)} chunks for {repo_url}.")


if __name__ == "__main__":
    from pathlib import Path
    from chunking import chunk_python_file

    test_file = Path("app/ingestion.py")
    chunks = chunk_python_file(test_file)
    index_chunks(chunks)