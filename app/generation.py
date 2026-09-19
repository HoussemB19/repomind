"""
RepoMind - Phase 2: generate an answer using DeepSeek V4 Pro via Lightning AI.
"""
from __future__ import annotations

import os
from openai import OpenAI
from dotenv import load_dotenv

try:
    from .indexing import get_model, get_collection
except ImportError:
    from indexing import get_model, get_collection

load_dotenv()

client = OpenAI(
    base_url="https://lightning.ai/api/v1",
    api_key=os.getenv("LIGHTNING_API_KEY"),
)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Build a prompt that forces the model to cite file:line."""
    context = "\n\n".join(
        f"File: {c['file']} (lines {c['start_line']}-{c['end_line']})\n```python\n{c['code']}\n```"
        for c in chunks
    )
    return f"""You are a code assistant. Answer the question using ONLY the code snippets below.
Always cite the exact file and line number in your answer, like `file.py:42`.

{context}

Question: {question}
Answer:"""


def ask(question: str, top_k: int = 3) -> str:
    model = get_model()
    collection = get_collection()

    question_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=top_k)

    chunks = [
        {
            "file": results["metadatas"][0][i]["file"],
            "start_line": results["metadatas"][0][i]["start_line"],
            "end_line": results["metadatas"][0][i]["end_line"],
            "code": results["documents"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]

    prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model="lightning-ai/deepseek-v4-pro",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    answer = ask("how does the code clone a git repository?")
    print(answer)