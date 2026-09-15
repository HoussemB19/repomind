"""
RepoMind - Phase 1 (suite): split Python files by function/class.
"""
from __future__ import annotations

import ast
from pathlib import Path


def chunk_python_file(file_path: Path) -> list[dict]:
    """Split a .py file into chunks, one per function or class."""
    source = file_path.read_text(encoding="utf-8", errors="ignore")
    chunks = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunks

    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno
            code = "\n".join(lines[start:end])
            chunks.append({
                "file": str(file_path),
                "name": node.name,
                "type": type(node).__name__,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "code": code,
            })

    return chunks


if __name__ == "__main__":
    # Quick manual test on this very file
    test_file = Path("app/ingestion.py")
    chunks = chunk_python_file(test_file)
    print(f"Found {len(chunks)} chunks in {test_file.name}:")
    for c in chunks:
        print(f"  - {c['type']} '{c['name']}' (lines {c['start_line']}-{c['end_line']})")