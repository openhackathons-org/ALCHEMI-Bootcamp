"""Checks for the NCI Atlas source-revision verification."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = BOOTCAMP_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_nci_subset import verify_source_checkout  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _clean_checkout(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "nci-atlas"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Tutorial Test")
    _git(repo, "config", "user.email", "tutorial-test@example.invalid")
    (repo / "source.txt").write_text("fixed source\n", encoding="utf-8")
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "-m", "source")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_source_checkout_accepts_the_exact_clean_revision(tmp_path: Path) -> None:
    repo, revision = _clean_checkout(tmp_path)

    assert verify_source_checkout(repo, expected_revision=revision) == revision


def test_source_checkout_rejects_a_different_revision(tmp_path: Path) -> None:
    repo, _ = _clean_checkout(tmp_path)

    with pytest.raises(ValueError, match="expected"):
        verify_source_checkout(repo, expected_revision="0" * 40)


def test_source_checkout_rejects_modified_tracked_inputs(tmp_path: Path) -> None:
    repo, revision = _clean_checkout(tmp_path)
    (repo / "source.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="modified tracked files"):
        verify_source_checkout(repo, expected_revision=revision)
