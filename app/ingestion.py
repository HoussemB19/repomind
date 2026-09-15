"""
RepoMind - Phase 1: Clone a GitHub repo and list its useful files.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import git

# Folders and file extensions we don't want to index
IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
USEFUL_EXTENSIONS = {".py", ".md", ".txt", ".js", ".ts", ".json", ".yaml", ".yml"}


def clone_repo(repo_url: str) -> Path:
    """Clone a GitHub repo into a temporary folder and return its path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="repomind_"))
    git.Repo.clone_from(repo_url, tmp_dir)
    return tmp_dir


def list_useful_files(repo_path: Path) -> list[Path]:
    """Walk the repo and return only files worth indexing."""
    useful_files = []
    for path in repo_path.rglob("*"):
        if path.is_file() and path.suffix in USEFUL_EXTENSIONS:
            if not any(ignored in path.parts for ignored in IGNORED_DIRS):
                useful_files.append(path)
    return useful_files


def cleanup(repo_path: Path):
    """Delete the temporary cloned repo folder."""
    shutil.rmtree(repo_path, ignore_errors=True)


if __name__ == "__main__":
    # Quick manual test — clone a small public repo and list its files
    test_url = "https://github.com/pallets/flask"
    print(f"Cloning {test_url}...")
    repo_path = clone_repo(test_url)
    files = list_useful_files(repo_path)
    print(f"Found {len(files)} useful files:")
    for f in files:
        print(f"  - {f.relative_to(repo_path)}")
    cleanup(repo_path)