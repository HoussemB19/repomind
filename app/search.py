"""
RepoMind - Phase 1 (final part): query ChromaDB with a natural language question.
"""
from __future__ import annotations

from indexing import get_model, get_collection


def search(question: str, top_k: int = 3):
    """Find the most relevant code chunks for a natural language question."""
    model = get_model()
    collection = get_collection()

    question_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
    )

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        print(f"\n--- Match {i+1} (distance: {distance:.4f}) ---")
        print(f"File: {metadata['file']}, Function: {metadata['name']} (line {metadata['start_line']})")
        print(results["documents"][0][i][:200])


if __name__ == "__main__":
    search("how does the code clone a git repository?")