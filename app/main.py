"""
RepoMind - Phase 3: expose ingestion + indexing + Q&A as a REST API.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

try:
    from .ingestion import clone_repo, list_useful_files, cleanup
    from .chunking import chunk_python_file
    from .indexing import index_chunks
    from .generation import ask
except ImportError:
    from ingestion import clone_repo, list_useful_files, cleanup
    from chunking import chunk_python_file
    from indexing import index_chunks
    from generation import ask

app = FastAPI(title="RepoMind", version="0.3.0")


class IndexRequest(BaseModel):
    repo_url: str


class QueryRequest(BaseModel):
    question: str
    repo_url: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "phase": "3 - API"}


@app.post("/index")
def index_repo(request: IndexRequest):
    """Clone a GitHub repo, chunk its Python files, and index them."""
    repo_path = clone_repo(request.repo_url)
    files = list_useful_files(repo_path)

    total_chunks = 0
    for file in files:
            if file.suffix == ".py":
                chunks = chunk_python_file(file)
                index_chunks(chunks, repo_url=request.repo_url)
                total_chunks += len(chunks)

    cleanup(repo_path)
    return {"repo": request.repo_url, "files_scanned": len(files), "chunks_indexed": total_chunks}


@app.post("/query")
def query_repo(request: QueryRequest):
    """Ask a natural language question about the indexed code."""
    answer = ask(request.question, repo_url=request.repo_url)
    return {"question": request.question, "answer": answer}