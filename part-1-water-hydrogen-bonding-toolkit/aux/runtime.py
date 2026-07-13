"""Portable repository and installed-package provenance checks."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path


def find_bootcamp_root(start: str | Path | None = None) -> Path:
    """Locate the Bootcamp root from any directory inside the checkout."""

    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "build" / "requirements.txt").is_file() and (
            candidate / "part-1-water-hydrogen-bonding-toolkit"
        ).is_dir():
            return candidate
    raise FileNotFoundError("Run this notebook inside the ALCHEMI-Bootcamp checkout.")


def installed_git_commit(distribution_name: str) -> str:
    """Read the immutable VCS revision recorded by Python packaging."""

    direct_url_text = metadata.distribution(distribution_name).read_text(
        "direct_url.json"
    )
    if direct_url_text is None:
        raise RuntimeError(f"{distribution_name} has no direct_url provenance")
    direct_url = json.loads(direct_url_text)
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if not commit:
        raise RuntimeError(f"{distribution_name} is not installed from a Git pin")
    return str(commit)


def verify_toolkit_pins(core_commit: str, ops_commit: str) -> dict[str, str]:
    """Fail when installed Toolkit revisions differ from the tutorial pins."""

    installed = {
        "Core": installed_git_commit("nvalchemi-toolkit"),
        "Ops": installed_git_commit("nvalchemi-toolkit-ops"),
    }
    expected = {"Core": str(core_commit), "Ops": str(ops_commit)}
    if installed != expected:
        raise RuntimeError(
            "Installed Toolkit pins do not match this notebook: "
            f"Core={installed['Core']}, Ops={installed['Ops']}"
        )
    return installed


__all__ = ["find_bootcamp_root", "installed_git_commit", "verify_toolkit_pins"]
