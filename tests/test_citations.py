from app.generation import add_missing_file_paths, validate_citations


def test_bare_line_reference_gets_an_unambiguous_file_path():
    chunks = [{"file": "src/app.py", "start_line": 300, "end_line": 330}]

    result = add_missing_file_paths("See lines 308-320.", chunks)

    assert result == "See src/app.py:308-320."


def test_ambiguous_bare_line_reference_is_not_guessed():
    chunks = [
        {"file": "src/app.py", "start_line": 300, "end_line": 330},
        {"file": "src/config.py", "start_line": 300, "end_line": 330},
    ]

    result = add_missing_file_paths("See lines 308-320.", chunks)

    assert result == "See lines 308-320."


def test_out_of_range_citation_gets_a_warning():
    chunks = [{"file": "src/app.py", "start_line": 300, "end_line": 330}]

    result = validate_citations("Evidence (src/app.py:500).", chunks)

    assert "Warning: unverified citation(s)" in result
