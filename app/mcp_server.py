"""
RepoMind - Phase 5: MCP server exposing RepoMind as a tool for Claude.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from mcp.server import MCPServer

mcp = MCPServer("RepoMind")


@mcp.tool()
def index_repo(repo_url: str) -> str:
    """Index a public GitHub repository so its code can be queried.
    Call this once per repository, and again after the repository changes,
    before using ask_repo. Takes about 30 seconds."""
    try:
        from .ingestion import clone_repo, list_useful_files, cleanup
        from .chunking import chunk_python_file
        from .indexing import index_chunks, clear_repo
    except ImportError:
        from ingestion import clone_repo, list_useful_files, cleanup
        from chunking import chunk_python_file
        from indexing import index_chunks, clear_repo

    repo_path = clone_repo(repo_url)
    files = list_useful_files(repo_path)

    all_chunks = []
    for file in files:
        if file.suffix == ".py":
            all_chunks.extend(chunk_python_file(file, root=repo_path))
    clear_repo(repo_url)
    for i in range(0, len(all_chunks), 256):
        index_chunks(all_chunks[i:i + 256], repo_url=repo_url)
    total_chunks = len(all_chunks)

    cleanup(repo_path)
    return f"Indexed {total_chunks} chunks from {len(files)} files in {repo_url}."


@mcp.tool()
def ask_repo(repo_url: str, question: str) -> str:
    """Answer a question about the code of a GitHub repository that was already
    indexed with index_repo. Use this for any question about how a repository's
    code works, even when the user only gives a GitHub URL. Do not clone or read
    the repository yourself. The answer cites file and line."""
    try:
        from .generation import ask
    except ImportError:
        from generation import ask

    return ask(question, repo_url=repo_url)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run(transport=transport)
    else:
        mcp.run(transport=transport, host="0.0.0.0", port=8001)
