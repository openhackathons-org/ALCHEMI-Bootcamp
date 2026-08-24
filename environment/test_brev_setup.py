# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable checks for Brev setup orchestration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REVISION = "a" * 40


def write_executable(path: Path, source: str) -> None:
    """Write one executable fixture command."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)


def prepare_brev_fixture(
    tmp_path: Path,
    *,
    failing_job: str = "",
) -> tuple[dict[str, str], Path, Path]:
    """Create a minimal course tree that records Brev setup job behavior."""
    course_root = tmp_path / "course"
    runtime_root = tmp_path / "runtime"
    setup_arguments = tmp_path / "setup-arguments.txt"
    job_events = tmp_path / "job-events.txt"
    fake_bin = tmp_path / "bin"
    course_root.mkdir()
    runtime_root.mkdir()
    (course_root / "uv.lock").write_text("fixture\n")

    write_executable(
        fake_bin / "git",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$ALCHEMI_TEST_REVISION\"\n",
    )
    write_executable(
        tmp_path / "uv",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == \"--version\" ]]; then\n"
        "    printf 'uv 0.12.1\\n'\n"
        "    exit 0\n"
        "fi\n"
        "printf 'Unexpected uv arguments: %s\\n' \"$*\" >&2\n"
        "exit 2\n",
    )
    write_executable(
        course_root / "scripts" / "setup",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" > \"$ALCHEMI_TEST_SETUP_ARGUMENTS\"\n"
        "[[ \"$*\" == \"--sync-only\" ]]\n"
        "[[ \"$ALCHEMI_TEST_FAILING_JOB\" != \"sync\" ]]\n",
    )
    write_executable(
        course_root / "scripts" / "run",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "[[ \"${1:-}\" == \"python\" ]]\n"
        "shift\n"
        "job=unknown\n"
        "device_file=\n"
        "case \"${1:-}\" in\n"
        "    -)\n"
        "        if [[ $# -ge 2 ]]; then\n"
        "            job=cuda\n"
        "            device_file=$2\n"
        "        else\n"
        "            job=assets-runtime\n"
        "        fi\n"
        "        ;;\n"
        "    *install_widget_frontend.py) job=widgets ;;\n"
        "esac\n"
        "printf 'start:%s\\n' \"$job\" >> \"$ALCHEMI_TEST_JOB_EVENTS\"\n"
        "sleep 0.2\n"
        "if [[ \"$job\" == \"$ALCHEMI_TEST_FAILING_JOB\" ]]; then\n"
        "    printf 'fail:%s\\n' \"$job\" >> \"$ALCHEMI_TEST_JOB_EVENTS\"\n"
        "    exit 23\n"
        "fi\n"
        "if [[ -n \"$device_file\" ]]; then\n"
        "    printf 'Fixture L40S\\n' > \"$device_file\"\n"
        "fi\n"
        "printf 'finish:%s\\n' \"$job\" >> \"$ALCHEMI_TEST_JOB_EVENTS\"\n",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "ALCHEMI_BREV_COURSE_ROOT": str(course_root),
            "ALCHEMI_BREV_EXPECTED_REVISION": EXPECTED_REVISION,
            "ALCHEMI_BREV_HOST_JUPYTER_DATA_DIR": str(tmp_path / "host-jupyter"),
            "ALCHEMI_BREV_KERNEL_ROOT": str(tmp_path / "kernels"),
            "ALCHEMI_BREV_SOURCE_ROOT": str(course_root),
            "ALCHEMI_BREV_WORKSPACE_ROOT": str(tmp_path / "workspace"),
            "ALCHEMI_RUNTIME_ROOT": str(runtime_root),
            "ALCHEMI_TEST_FAILING_JOB": failing_job,
            "ALCHEMI_TEST_JOB_EVENTS": str(job_events),
            "ALCHEMI_TEST_REVISION": EXPECTED_REVISION,
            "ALCHEMI_TEST_SETUP_ARGUMENTS": str(setup_arguments),
            "ALCHEMI_UV_BIN": str(tmp_path / "uv"),
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
        }
    )
    return environment, runtime_root, setup_arguments


def run_brev_setup(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the repository Brev setup script against the fixture tree."""
    return subprocess.run(
        [str(ROOT / "scripts" / "brev-setup.sh")],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_brev_setup_runs_post_sync_jobs_concurrently(tmp_path: Path) -> None:
    environment, runtime_root, setup_arguments = prepare_brev_fixture(tmp_path)

    completed = run_brev_setup(environment)

    assert completed.returncode == 0, completed.stderr
    assert setup_arguments.read_text().strip() == "--sync-only"
    events = Path(environment["ALCHEMI_TEST_JOB_EVENTS"]).read_text().splitlines()
    starts = [events.index(f"start:{job}") for job in ("assets-runtime", "widgets", "cuda")]
    finishes = [
        events.index(f"finish:{job}") for job in ("assets-runtime", "widgets", "cuda")
    ]
    assert max(starts) < min(finishes)

    ready = (runtime_root / "brev-ready.txt").read_text()
    assert "status=ready\n" in ready
    assert "gpu=Fixture L40S\n" in ready
    assert not list(runtime_root.glob("brev-ready.txt.tmp.*"))


def test_brev_setup_waits_for_jobs_and_clears_stale_ready_file(tmp_path: Path) -> None:
    environment, runtime_root, _ = prepare_brev_fixture(
        tmp_path,
        failing_job="widgets",
    )
    ready_file = runtime_root / "brev-ready.txt"
    ready_file.write_text("status=ready\nrevision=stale\n")

    completed = run_brev_setup(environment)

    assert completed.returncode != 0
    assert "Brev setup job failed: widgets" in completed.stderr
    assert not ready_file.exists()
    events = Path(environment["ALCHEMI_TEST_JOB_EVENTS"]).read_text().splitlines()
    assert "fail:widgets" in events
    assert "finish:assets-runtime" in events
    assert "finish:cuda" in events


def test_brev_setup_clears_stale_ready_before_sync(tmp_path: Path) -> None:
    environment, runtime_root, _ = prepare_brev_fixture(
        tmp_path,
        failing_job="sync",
    )
    ready_file = runtime_root / "brev-ready.txt"
    ready_file.write_text("status=ready\nrevision=stale\n")

    completed = run_brev_setup(environment)

    assert completed.returncode != 0
    assert not ready_file.exists()
    assert not Path(environment["ALCHEMI_TEST_JOB_EVENTS"]).exists()
