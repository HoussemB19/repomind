from pathlib import Path

from app.chunking import chunk_python_file


def test_chunks_functions_and_classes(tmp_path: Path):
    """One chunk should be produced per function and per class."""
    source = '''
def add(a, b):
    return a + b


class Greeter:
    def hello(self):
        return "hi"
'''
    file = tmp_path / "sample.py"
    file.write_text(source)

    chunks = chunk_python_file(file)
    names = {c["name"] for c in chunks}

    assert names == {"add", "Greeter", "hello"}


def test_chunk_has_correct_metadata(tmp_path: Path):
    """Each chunk should record its type and exact line range."""
    source = "def add(a, b):\n    return a + b\n"
    file = tmp_path / "sample.py"
    file.write_text(source)

    chunks = chunk_python_file(file)

    assert len(chunks) == 1
    assert chunks[0]["type"] == "FunctionDef"
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["end_line"] == 2
    assert "return a + b" in chunks[0]["code"]


def test_syntax_error_returns_empty_list(tmp_path: Path):
    """A broken file must not crash indexing; it should just be skipped."""
    file = tmp_path / "broken.py"
    file.write_text("def broken(:\n")

    assert chunk_python_file(file) == []
def test_file_path_is_relative_to_root(tmp_path: Path):
    """Chunks must cite repo-relative paths, not temp clone paths."""
    pkg = tmp_path / "app"
    pkg.mkdir()
    file = pkg / "sample.py"
    file.write_text("def add(a, b):\n    return a + b\n")

    chunks = chunk_python_file(file, root=tmp_path)

    assert chunks[0]["file"] == "app/sample.py"   