# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the reproducible setup recipe."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_setup_selects_the_pinned_managed_python(tmp_path: Path) -> None:
    call_log = tmp_path / "uv-calls.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$ALCHEMI_TEST_UV_CALLS\"\n"
    )
    fake_uv.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "ALCHEMI_INSTALL_PROFILE": "runtime",
            "ALCHEMI_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "ALCHEMI_TEST_UV_CALLS": str(call_log),
            "ALCHEMI_UV_BIN": str(fake_uv),
        }
    )
    subprocess.run(
        [str(ROOT / "scripts" / "setup")],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    python_version = (ROOT / ".python-version").read_text().strip()
    sync_call, run_call = call_log.read_text().splitlines()
    expected = f"--managed-python --python {python_version}"
    assert expected in sync_call
    assert expected in run_call


def test_setup_sync_only_skips_asset_prewarming(tmp_path: Path) -> None:
    call_log = tmp_path / "uv-calls.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$ALCHEMI_TEST_UV_CALLS\"\n"
    )
    fake_uv.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "ALCHEMI_INSTALL_PROFILE": "brev",
            "ALCHEMI_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "ALCHEMI_TEST_UV_CALLS": str(call_log),
            "ALCHEMI_UV_BIN": str(fake_uv),
        }
    )
    subprocess.run(
        [str(ROOT / "scripts" / "setup"), "--sync-only"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = call_log.read_text().splitlines()
    assert len(calls) == 1
    assert calls[0].startswith("sync ")


def test_setup_rejects_extra_arguments() -> None:
    completed = subprocess.run(
        [str(ROOT / "scripts" / "setup"), "--sync-only", "unexpected"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Usage:" in completed.stderr
