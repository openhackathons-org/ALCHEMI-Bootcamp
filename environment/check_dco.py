# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require one Developer Certificate of Origin sign-off per commit."""

from __future__ import annotations

import os
import re
import subprocess

SIGNOFF = re.compile(r"(?m)^Signed-off-by: .+ <[^<>\s]+@[^<>\s]+>$")


def release_range(environment: dict[str, str]) -> str:
    """Return the Git revision range introduced by the current GitHub event."""
    if environment.get("GITHUB_EVENT_NAME") == "pull_request":
        base_ref = environment.get("GITHUB_BASE_REF", "")
        if not base_ref:
            raise RuntimeError("GITHUB_BASE_REF is required for a pull request")
        return f"origin/{base_ref}..HEAD"

    before = environment.get("GITHUB_EVENT_BEFORE", "")
    if before and before != "0" * 40:
        return f"{before}..HEAD"
    return "HEAD^..HEAD"


def unsigned_commits(log_output: str) -> list[str]:
    """Return commit IDs whose messages lack a DCO sign-off."""
    records = log_output.split("\x1e")
    unsigned = []
    for record in records:
        if not record.strip():
            continue
        commit, _, message = record.partition("\x1f")
        if not SIGNOFF.search(message):
            unsigned.append(commit.strip())
    return unsigned


def main() -> None:
    """Check commits introduced by the current GitHub event."""
    revision_range = release_range(dict(os.environ))
    completed = subprocess.run(
        ["git", "log", "--format=%H%x1f%B%x1e", revision_range],
        check=True,
        capture_output=True,
        text=True,
    )
    unsigned = unsigned_commits(completed.stdout)
    if unsigned:
        formatted = "\n".join(f"  {commit}" for commit in unsigned)
        raise RuntimeError(f"Commits missing Signed-off-by:\n{formatted}")
    print(f"DCO sign-offs verified for {revision_range}")


if __name__ == "__main__":
    main()
