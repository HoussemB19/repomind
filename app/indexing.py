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
    """Get or create a ChromaDB collection (server mode if CHROMA_HOST is set)."""
    host = os.getenv("CHROMA_HOST")
    if host:
        client = chromadb.HttpClient(host=host, port=int(os.getenv("CHROMA_PORT", "8000")))
    else:
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
    embed_inputs = [f"# file: {c['file']}\n# {c['type']}: {c['name']}\n{c['code']}" for c in chunks]
    embeddings = model.encode(embed_inputs).tolist()

    ids = [f"{repo_url}::{c['file']}::{c['name']}::{c['start_line']}" for c in chunks]
    metadatas = [
        {"file": c["file"], "name": c["name"], "type": c["type"],
         "start_line": c["start_line"], "end_line": c["end_line"], "repo": repo_url}
        for c in chunks
    ]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Indexed {len(chunks)} chunks for {repo_url}.")

def clear_repo(repo_url: str) -> None:
    """Delete all chunks previously indexed for this repo."""
    get_collection().delete(where={"repo": repo_url})
if __name__ == "__main__":
    from pathlib import Path
    from chunking import chunk_python_file

    test_file = Path("app/ingestion.py")
    chunks = chunk_python_file(test_file)
    index_chunks(chunks)