"""Small notebook-runtime helpers for the adsorption tutorial."""

from __future__ import annotations

from pathlib import Path


def make_tutorial_relpath(tutorial_root: str | Path):
    """Return a display helper that keeps paths relative to the tutorial folder."""
    root = Path(tutorial_root).resolve()

    def tutorial_relpath(path: str | Path) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            return candidate.resolve().relative_to(root).as_posix()
        except ValueError:
            return candidate.name

    return tutorial_relpath
