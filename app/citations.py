"""Citation normalization and validation helpers."""

from __future__ import annotations

import re


def add_missing_file_paths(answer: str, chunks: list[dict]) -> str:
    """Add file paths to line references when the source range is unambiguous."""
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

    def replace_bare(match):
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

    return bare_pattern.sub(replace_bare, answer)


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
