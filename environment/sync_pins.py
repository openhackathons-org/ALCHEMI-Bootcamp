# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy the pins in runtime-pins.toml into the files that repeat them.

`environment/runtime-pins.toml` is the one place to change a pin. To move the
tutorials to a new Toolkit or Toolkit-Ops commit:

1. Edit the `[toolkit]` or `[toolkit_ops]` commit, version and dates.
2. Run `python environment/sync_pins.py` to update `pyproject.toml`,
   `THIRD_PARTY_NOTICES.md` and `SOURCES_AND_LICENSES.md`.
3. Run `uv lock` with the uv version in `scripts/brev-setup.sh`.
4. Run `./scripts/setup`, the tests, and `.licenses/generate_licenses.sh`.

`python environment/sync_pins.py --check` reports any file that disagrees with
the pins, including `uv.lock`, and changes nothing.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PINS_FILE = ROOT / "environment" / "runtime-pins.toml"

Rule = tuple[str, str, Callable[[Mapping[str, Any]], str]]


def _github_path(repository: str) -> str:
    """Return the owner/name path of a GitHub repository URL."""
    match = re.fullmatch(r"https://github\.com/(.+?)(?:\.git)?", repository)
    if match is None:
        raise ValueError(f"Expected a GitHub repository URL: {repository}")
    return match.group(1)


def _requirement(section: str, extra: str) -> Callable[[Mapping[str, Any]], str]:
    def render(pins: Mapping[str, Any]) -> str:
        pin = pins[section]
        return (
            f'"{pin["distribution"]}[{extra}] @ git+{pin["repository"]}'
            f'@{pin["commit"]}"'
        )

    return render


def _license_link(section: str) -> Callable[[Mapping[str, Any]], str]:
    def render(pins: Mapping[str, Any]) -> str:
        pin = pins[section]
        path = _github_path(pin["repository"])
        return f"https://github.com/{path}/blob/{pin['commit']}/LICENSE"

    return render


def _exact(name: str, value: Callable[[Mapping[str, Any]], str]) -> Rule:
    pattern = rf'"{re.escape(name)}==[^"]+"'
    return pattern, name, lambda pins: f'"{name}=={value(pins)}"'


def _toolkit_summary(pins: Mapping[str, Any]) -> str:
    toolkit = pins["toolkit"]
    ops = pins["toolkit_ops"]
    return (
        f"- NVIDIA ALCHEMI Toolkit {toolkit['version']} (commit "
        f"`{toolkit['commit'][:7]}`) and Toolkit-Ops {ops['version']} "
        f"(commit `{ops['commit'][:7]}`): Apache-2.0."
    )


RULES: dict[str, list[Rule]] = {
    "pyproject.toml": [
        (
            r'"nvalchemi-toolkit\[mace\] @ git\+[^"]+"',
            "Toolkit requirement",
            _requirement("toolkit", "mace"),
        ),
        (
            r'"nvalchemi-toolkit-ops\[torch-cu13\] @ git\+[^"]+"',
            "Toolkit-Ops requirement",
            _requirement("toolkit_ops", "torch-cu13"),
        ),
        _exact("torch", lambda pins: pins["torch"]),
        _exact("ase", lambda pins: pins["ase"]),
        _exact(
            "aimnet",
            lambda pins: pins["model"]["package_version"],
        ),
        _exact(
            "mace-torch",
            lambda pins: pins["mace"]["package_version"],
        ),
        _exact("rdkit", lambda pins: pins["rdkit"]["package_version"]),
        _exact(
            "py3dmol",
            lambda pins: pins["structure_viewer"]["package_version"],
        ),
    ],
    "THIRD_PARTY_NOTICES.md": [
        (
            r"https://github\.com/NVIDIA/nvalchemi-toolkit/blob/[0-9a-f]{40}/LICENSE",
            "Toolkit license link",
            _license_link("toolkit"),
        ),
        (
            r"https://github\.com/NVIDIA/nvalchemi-toolkit-ops/blob/[0-9a-f]{40}/LICENSE",
            "Toolkit-Ops license link",
            _license_link("toolkit_ops"),
        ),
        (
            r"The Core modules use `aimnet [^`]+`",
            "AIMNet version",
            lambda pins: (
                f"The Core modules use `aimnet {pins['model']['package_version']}`"
            ),
        ),
    ],
    "SOURCES_AND_LICENSES.md": [
        (
            r"- NVIDIA ALCHEMI Toolkit [^\n]*Toolkit-Ops [^\n]*: Apache-2\.0\.",
            "Toolkit summary",
            _toolkit_summary,
        ),
    ],
}


def load_pins(path: Path = PINS_FILE) -> dict[str, Any]:
    """Read the runtime pins."""
    return tomllib.loads(path.read_text())


def apply_rules(
    text: str, rules: list[Rule], pins: Mapping[str, Any], file_name: str
) -> tuple[str, list[str]]:
    """Return updated text and one message per value that changed."""
    changes = []
    for pattern, label, render in rules:
        expected = render(pins)
        matches = list(re.finditer(pattern, text))
        if len(matches) != 1:
            raise RuntimeError(
                f"{file_name}: expected one {label}, found {len(matches)}"
            )
        if matches[0].group(0) != expected:
            changes.append(f"{file_name}: {label} -> {expected}")
            text = text[: matches[0].start()] + expected + text[matches[0].end() :]
    return text, changes


def lock_mismatches(lock_text: str, pins: Mapping[str, Any]) -> list[str]:
    """Return messages for Toolkit packages whose lock entry differs from the pins."""
    lock = tomllib.loads(lock_text)
    packages = {package["name"]: package for package in lock.get("package", [])}
    problems = []
    for section in ("toolkit", "toolkit_ops"):
        pin = pins[section]
        package = packages.get(pin["distribution"])
        if package is None:
            problems.append(f"uv.lock: {pin['distribution']} is missing")
            continue
        source = package.get("source", {}).get("git", "")
        if not source.endswith(f"#{pin['commit']}"):
            problems.append(
                f"uv.lock: {pin['distribution']} source {source} "
                f"is not commit {pin['commit']}; run uv lock"
            )
        if package.get("version") != pin["version"]:
            problems.append(
                f"uv.lock: {pin['distribution']} version {package.get('version')} "
                f"differs from {pin['version']}"
            )
    return problems


def sync(root: Path = ROOT, *, check: bool) -> list[str]:
    """Update or check every file that repeats a pin."""
    pins = load_pins(root / "environment" / "runtime-pins.toml")
    messages = []
    for file_name, rules in RULES.items():
        path = root / file_name
        text = path.read_text()
        updated, changes = apply_rules(text, rules, pins, file_name)
        messages.extend(changes)
        if changes and not check:
            path.write_text(updated)
    if check:
        messages.extend(lock_mismatches((root / "uv.lock").read_text(), pins))
    return messages


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report files that differ from runtime-pins.toml and change nothing",
    )
    arguments = parser.parse_args(argv)
    messages = sync(check=arguments.check)
    for message in messages:
        print(message)
    if arguments.check and messages:
        return 1
    if not messages:
        print("All repeated pins match environment/runtime-pins.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
