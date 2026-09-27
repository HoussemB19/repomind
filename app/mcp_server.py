"""
RepoMind - Phase 5: MCP server exposing RepoMind as a tool for Claude.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
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

mcp = MCPServer("RepoMind")

@mcp.tool()
def index_repo(repo_url: str) -> str:
    """Clone a GitHub repository and index its Python files for question-answering."""
    repo_path = clone_repo(repo_url)
    files = list_useful_files(repo_path)

    total_chunks = 0
    for file in files:
        if file.suffix == ".py":
            chunks = chunk_python_file(file)
            index_chunks(chunks, repo_url=repo_url)
            total_chunks += len(chunks)

    cleanup(repo_path)
    return f"Indexed {total_chunks} chunks from {len(files)} files in {repo_url}."


@mcp.tool()
def ask_repo(repo_url: str, question: str) -> str:
    """Ask a natural language question about a previously indexed GitHub repository."""
    return ask(question, repo_url=repo_url)


if __name__ == "__main__":
    mcp.run()