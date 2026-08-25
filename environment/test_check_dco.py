# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for release commit sign-off validation."""

from __future__ import annotations

import pytest
from check_dco import release_range, unsigned_commits


def test_release_range_uses_pull_request_base() -> None:
    environment = {
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_BASE_REF": "main",
    }

    assert release_range(environment) == "origin/main..HEAD"


def test_release_range_uses_push_before_commit() -> None:
    environment = {
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_EVENT_BEFORE": "a" * 40,
    }

    assert release_range(environment) == f"{'a' * 40}..HEAD"


def test_release_range_checks_tip_when_push_has_no_before_commit() -> None:
    environment = {
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_EVENT_BEFORE": "0" * 40,
    }

    assert release_range(environment) == "HEAD^..HEAD"


def test_release_range_rejects_missing_pull_request_base() -> None:
    with pytest.raises(RuntimeError, match="GITHUB_BASE_REF"):
        release_range({"GITHUB_EVENT_NAME": "pull_request"})


def test_unsigned_commits_reports_only_missing_signoff() -> None:
    log_output = (
        "a" * 40
        + "\x1fRelease setup\n\nSigned-off-by: User Name <user@example.com>\n\x1e"
        + "b" * 40
        + "\x1fUnsigned change\n\x1e"
    )

    assert unsigned_commits(log_output) == ["b" * 40]
