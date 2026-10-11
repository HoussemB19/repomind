"""RepoMind - Phase 2: generate answers using an LLM."""

from __future__ import annotations
import re
import os
from openai import APITimeoutError, OpenAI
from dotenv import load_dotenv

try:
    from .indexing import get_model, get_collection
except ImportError:
    from indexing import get_model, get_collection

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    timeout=90.0,
    max_retries=1,
)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Build a prompt that forces the model to cite file:line."""
    def numbered(c):
        return "\n".join(
            f"{c['start_line'] + i}: {line}"
            for i, line in enumerate(c["code"].splitlines())
        )

    context = "\n\n".join(
        f"File: {c['file']} (lines {c['start_line']}-{c['end_line']})\n```python\n{numbered(c)}\n```"
        for c in chunks
    )
    return f"""You are a code assistant. Answer the question using ONLY the code snippets below.
For every factual claim, cite the exact source line(s) that directly support that claim,
immediately after the claim, like `file.py:42`. Do not cite a whole enclosing function
or unrelated import lines when a more specific line contains the evidence. Use a line
  range only when every line in the range supports the same claim. If the snippets do not
  support a claim, say that the information is not present instead of guessing. A citation
  without a file path, such as `lines 10-12`, is invalid; always use `path/to/file.py:10-12`.
  When useful,
  include a short code excerpt alongside the citation so the evidence is visible.
Describe behavior from the actual function body, not from a function name or guess. Distinguish
side effects such as deleting database records, deleting files, or cleaning temporary resources
only when the cited code directly shows that behavior. Do not assume fixed file paths, function
names, frameworks, or repository structure; infer them from the supplied snippets.

{context}

Question: {question}
    Answer:"""


def add_missing_file_paths(answer: str, chunks: list[dict]) -> str:
    """Add file paths to bare line references when the source range is unambiguous."""
    explicit_pattern = re.compile(
        r"\blines?[\s\u00a0\u202f]+(?P<start>\d+)"
        r"(?:[\-\u2010\u2011\u2012\u2013\u2212](?P<end>\d+))?"
        r"\s+of\s+[`*_]*(?P<file>[A-Za-z0-9_./\\-]+\.py)[`*_]*"
    )

    def replace_explicit(match):
        start = match.group("start")
        end = match.group("end")
        file = match.group("file")
        return f"{file}:{start}-{end}" if end else f"{file}:{start}"

    answer = explicit_pattern.sub(replace_explicit, answer)

    bare_pattern = re.compile(
        r"\blines?[\s\u00a0\u202f]+(?P<start>\d+)"
        r"(?:[\-\u2010\u2011\u2012\u2013\u2212](?P<end>\d+))?"
    )

    def replace(match):
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        candidates = {
            chunk["file"]
            for chunk in chunks
            if start >= chunk["start_line"] and end <= chunk["end_line"]
        }
        if len(candidates) == 1:
            file = next(iter(candidates))
            return f"{file}:{start}-{end}" if end != start else f"{file}:{start}"
        return match.group(0)

    return bare_pattern.sub(replace, answer)


def validate_citations(answer: str, chunks: list[dict]) -> str:
    """Append a warning when the model cites lines outside retrieved chunks."""
    citation_pattern = re.compile(
        r"(?P<file>[A-Za-z0-9_./\\-]+\.py):(?P<start>\d+)"
        r"(?:-(?P<end>\d+))?"
    )
    ranges = {}
    for chunk in chunks:
        ranges.setdefault(chunk["file"], []).append(
            (chunk["start_line"], chunk["end_line"])
        )

    invalid = []
    normalized_answer = answer.replace("\\_", "_")
    for match in citation_pattern.finditer(normalized_answer):
        file = match.group("file")
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if not any(
            file == known_file and start >= first and end <= last
            for known_file, file_ranges in ranges.items()
            for first, last in file_ranges
        ):
            invalid.append(match.group(0))

    if invalid:
        unique_invalid = ", ".join(dict.fromkeys(invalid))
        return (
            f"{answer}\n\n[Warning: unverified citation(s): "
            f"{unique_invalid}]"
        )
    return answer


def ask(question: str, repo_url: str | None = None, top_k: int = 5) -> str:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    model = get_model()
    collection = get_collection()

    question_embedding = model.encode([question]).tolist()

    where_filter = {"repo": repo_url} if repo_url else None
    n_results = top_k

    if repo_url:
        stored = collection.get(where={"repo": repo_url}, include=["metadatas"])
        if not stored["ids"]:
            return f"{repo_url} is not indexed yet. Call index_repo first."
        q = question.lower()
        named = {
            m["file"]
            for m in stored["metadatas"]
            if re.search(r"(?<![\w.])" + re.escape(m["file"].split("/")[-1].lower()) + r"(?![\w])", q)
        }
        if named:
            where_filter = {"$and": [{"repo": repo_url}, {"file": {"$in": list(named)}}]}
            n_results = min(15, sum(1 for m in stored["metadatas"] if m["file"] in named))

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results,
        where=where_filter,
    )

    if not results.get("ids") or not results["ids"] or not results["ids"][0]:
        if repo_url:
            return f"No indexed code chunks were found for {repo_url}."
        return "No indexed code chunks were found. Index a repository first."

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

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except APITimeoutError:
        return "The LLM request timed out after 90 seconds. Please try again."
    answer = response.choices[0].message.content or ""
    answer = add_missing_file_paths(answer, chunks)
    return validate_citations(answer, chunks)


if __name__ == "__main__":
    answer = ask("how does the code clone a git repository?")
    print(answer)
