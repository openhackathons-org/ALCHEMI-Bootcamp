"""Static and Git-backed checks for the Part 1 H100 launch path."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CHECK_PATH = REPOSITORY_ROOT / "build" / "verify_part1_runtime.py"
RUN_VALIDATOR_PATH = REPOSITORY_ROOT / "scripts" / "validate_part1_ir_run.py"
ENVIRONMENT_PATH = REPOSITORY_ROOT / "build" / "environment.yml"
SETUP_SBATCH_PATH = REPOSITORY_ROOT / "scripts" / "slurm_part1_sevennet_setup.sbatch"
DOMAIN_SBATCH_PATH = (
    REPOSITORY_ROOT / "scripts" / "slurm_part1_domain_decomposition.sbatch"
)
DOMAIN_LAUNCHER_PATH = REPOSITORY_ROOT / "scripts" / "run_part1_domain_decomposition.sh"
DOMAIN_PLAN_PATH = REPOSITORY_ROOT / "scripts" / "part1_domain_plan.py"
REMASTER_SBATCH_PATH = REPOSITORY_ROOT / "scripts" / "slurm_part1_remaster_h100.sbatch"
COMPUTE_LAB_RUNBOOK_PATH = (
    REPOSITORY_ROOT / "part-1-scalable-atomistic-workflows" / "COMPUTE_LAB_RUNBOOK.md"
)
DOMAIN_RESULTS_PATH = (
    REPOSITORY_ROOT
    / "part-1-scalable-atomistic-workflows"
    / "aux"
    / "domain"
    / "results.py"
)
SBATCH_NAMES = (
    "slurm_part1_domain_decomposition.sbatch",
    "slurm_part1_remaster_h100.sbatch",
    "slurm_part1_sevennet_setup.sbatch",
)
RETIRED_ORBMOL_TESTS = (
    "test_orbmol_adapter.py",
    "test_orbmol_checks.py",
    "test_orbmol_config.py",
)
PYTHON_LAUNCH_ANCHORS = (
    (
        "scripts/slurm_part1_domain_decomposition.sbatch",
        "mapfile -t DOMAIN_METHODOLOGY_DEFAULTS",
    ),
    (
        "scripts/slurm_part1_remaster_h100.sbatch",
        '"$PYTHON" -m ipykernel install',
    ),
    (
        "scripts/slurm_part1_sevennet_setup.sbatch",
        '"$MAIN_ENV/bin/python" - "$MAIN_ENV/conda-meta"',
    ),
    (
        "scripts/run_part1_domain_decomposition.sh",
        'exec "$TORCHRUN"',
    ),
)


def _load_runtime_check() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "part1_runtime_check",
        RUNTIME_CHECK_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_run_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "part1_run_validator",
        RUN_VALIDATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNTIME_CHECK = _load_runtime_check()
RUN_VALIDATOR = _load_run_validator()


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _run_launcher_network_mode(
    tmp_path: Path,
    *,
    ip_script: str,
    mode: str,
    environment: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    ip = fake_bin / "ip"
    ip.write_text(ip_script, encoding="utf-8")
    ip.chmod(0o755)

    run_environment = os.environ.copy()
    run_environment["PATH"] = f"{fake_bin}:{run_environment['PATH']}"
    for name in (
        "ALCHEMI_DISTRIBUTED_IFACE",
        "ALCHEMI_MASTER_ADDR",
    ):
        run_environment.pop(name, None)
    if environment is not None:
        run_environment.update(environment)

    return subprocess.run(
        ["bash", str(DOMAIN_LAUNCHER_PATH), mode],
        check=check,
        capture_output=True,
        text=True,
        env=run_environment,
    )


def _clean_test_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "--quiet")
    for relative_path in RUNTIME_CHECK.REQUIRED_TRACKED_SOURCE_PATHS:
        path = checkout / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative_path}\n", encoding="utf-8")
    (checkout / RUNTIME_CHECK.SOURCE_MANIFEST_RELATIVE_PATH).write_text(
        (REPOSITORY_ROOT / RUNTIME_CHECK.SOURCE_MANIFEST_RELATIVE_PATH).read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    (checkout / ".gitignore").write_text("ignored-output.txt\n", encoding="utf-8")
    _git(checkout, "add", "--all")
    _git(
        checkout,
        "-c",
        "user.name=Part 1 test",
        "-c",
        "user.email=part1@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "--message=tracked source",
    )
    return checkout


def test_runtime_check_requires_the_declared_h100_environment() -> None:
    source = RUNTIME_CHECK_PATH.read_text(encoding="utf-8")

    assert RUNTIME_CHECK.EXPECTED_PYTHON == (3, 12, 13)
    assert RUNTIME_CHECK.EXPECTED_TORCH_VERSION == "2.12.0+cu130"
    assert RUNTIME_CHECK.EXPECTED_CUDA_VERSION == "13.0"
    for check in (
        "sys.version_info[:3] != EXPECTED_PYTHON",
        "torch_distribution_version != EXPECTED_TORCH_VERSION",
        "torch.__version__ != EXPECTED_TORCH_VERSION",
        "torch.version.cuda != EXPECTED_CUDA_VERSION",
        '"H100" not in cuda_device.upper()',
        "uv_version_tuple < (0, 9, 26)",
        '"resolved_scientific_versions": resolved_scientific_versions',
    ):
        assert check in source
    assert set(RUNTIME_CHECK.RECORDED_SCIENTIFIC_VERSIONS) == {
        "ase",
        "matscipy",
        "numpy",
        "pandas",
        "pydantic",
        "pymatgen",
        "scipy",
        "torch-geometric",
        "triton",
    }


def test_conda_spec_has_the_python_patch_and_required_uv_features() -> None:
    source = ENVIRONMENT_PATH.read_text(encoding="utf-8")

    assert "  - python=3.12.13\n" in source
    assert "  - uv>=0.9.26\n" in source
    assert "  - python=3.12\n" not in source
    assert "  - uv\n" not in source


def test_runtime_check_resolves_packmol_from_the_explicit_base_environment(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "python-overlay"
    base = tmp_path / "conda-base"
    overlay.mkdir()
    base.mkdir()

    assert RUNTIME_CHECK.resolve_base_environment(overlay, base) == base.resolve()
    assert RUNTIME_CHECK.resolve_base_environment(overlay, None) == overlay.resolve()
    with pytest.raises(RuntimeError, match="base environment does not exist"):
        RUNTIME_CHECK.resolve_base_environment(overlay, tmp_path / "missing")


def test_clean_source_check_rejects_untracked_files_and_tracked_edits(
    tmp_path: Path,
) -> None:
    checkout = _clean_test_checkout(tmp_path)

    RUNTIME_CHECK.verify_clean_tracked_source(checkout)
    (checkout / "untracked-output.txt").write_text("output\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked or untracked"):
        RUNTIME_CHECK.verify_clean_tracked_source(checkout)
    (checkout / "untracked-output.txt").unlink()

    (checkout / "ignored-output.txt").write_text("ignored\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ignored files"):
        RUNTIME_CHECK.verify_clean_tracked_source(checkout)
    (checkout / "ignored-output.txt").unlink()

    tracked = checkout / RUNTIME_CHECK.REQUIRED_TRACKED_SOURCE_PATHS[0]
    tracked.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked or untracked"):
        RUNTIME_CHECK.verify_clean_tracked_source(checkout)


def test_clean_runtime_source_report_matches_packaged_validator(
    tmp_path: Path,
) -> None:
    checkout = _clean_test_checkout(tmp_path)
    report = RUNTIME_CHECK.source_report(
        checkout,
        require_clean_source=True,
        skip_source_check=False,
    )

    validated = RUN_VALIDATOR.validate_packaged_source_identity(
        report,
        source_root=checkout,
    )

    assert validated == report


def test_source_manifest_covers_local_part1_document_links() -> None:
    part1_root = REPOSITORY_ROOT / "part-1-scalable-atomistic-workflows"
    notebook = json.loads(
        (part1_root / "alchemi-water-ir.ipynb").read_text(encoding="utf-8")
    )
    documents = [
        (part1_root / "README.md", (part1_root / "README.md").read_text()),
        *(
            (
                part1_root / "alchemi-water-ir.ipynb",
                "".join(cell.get("source", [])),
            )
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        ),
    ]
    linked_files: set[str] = set()
    for source_path, text in documents:
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            target_path = (source_path.parent / target.split("#", 1)[0]).resolve()
            if target_path.is_dir():
                target_path = target_path / "README.md"
            if not target_path.is_file():
                continue
            linked_files.add(target_path.relative_to(REPOSITORY_ROOT).as_posix())

    assert linked_files <= set(RUNTIME_CHECK.REQUIRED_TRACKED_SOURCE_PATHS)


@pytest.mark.parametrize("name", SBATCH_NAMES)
def test_h100_launch_examples_keep_scheduler_logs_outside_the_checkout(
    name: str,
) -> None:
    source = (REPOSITORY_ROOT / "scripts" / name).read_text(encoding="utf-8")

    assert "#SBATCH --output" not in source
    assert '--chdir="$ALCHEMI_RUN_ROOT/logs"' in source
    assert '--output="$ALCHEMI_RUN_ROOT/logs/' in source
    assert f'"$ALCHEMI_SHARED_REPO/scripts/{name}"' in source


@pytest.mark.parametrize(("relative_path", "first_python"), PYTHON_LAUNCH_ANCHORS)
def test_h100_launchers_disable_bytecode_before_starting_python(
    relative_path: str,
    first_python: str,
) -> None:
    source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    setting = "export PYTHONDONTWRITEBYTECODE=1"

    assert source.count(setting) == 1
    assert source.index(setting) < source.index(first_python)


def test_domain_launcher_uses_the_requested_local_process_count() -> None:
    source = DOMAIN_LAUNCHER_PATH.read_text(encoding="utf-8")

    assert (
        ': "${ALCHEMI_DOMAIN_GPUS:?set ALCHEMI_DOMAIN_GPUS for the local launch}"'
        in source
    )
    assert '--nproc-per-node "$ALCHEMI_DOMAIN_GPUS"' in source
    assert 'case "$ALCHEMI_DOMAIN_GPUS"' not in source


def test_domain_launcher_resolves_the_automatic_master_address(
    tmp_path: Path,
) -> None:
    result = _run_launcher_network_mode(
        tmp_path,
        mode="--print-master-address",
        ip_script="""#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  "-4 route get 1.1.1.1")
    echo "1.1.1.1 via 10.117.0.1 dev eno1np0 src 10.117.10.97"
    ;;
  "-4 -o addr show dev eno1np0 scope global")
    echo "2: eno1np0 inet 10.117.10.97/19 scope global eno1np0"
    ;;
  *)
    echo "unexpected ip arguments: $*" >&2
    exit 2
    ;;
esac
""",
    )

    assert result.stdout.strip() == "10.117.10.97"


def test_domain_launcher_uses_an_explicit_master_interface_without_public_routing(
    tmp_path: Path,
) -> None:
    result = _run_launcher_network_mode(
        tmp_path,
        mode="--print-master-address",
        environment={"ALCHEMI_DISTRIBUTED_IFACE": "cluster0"},
        ip_script="""#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  "link show dev cluster0")
    exit 0
    ;;
  "-4 -o addr show dev cluster0 scope global")
    echo "2: cluster0 inet 10.90.0.12/24 scope global cluster0"
    ;;
  *)
    echo "unexpected ip arguments: $*" >&2
    exit 2
    ;;
esac
""",
    )

    assert result.stdout.strip() == "10.90.0.12"


def test_domain_launcher_maps_the_master_address_to_its_non_loopback_device(
    tmp_path: Path,
) -> None:
    result = _run_launcher_network_mode(
        tmp_path,
        mode="--print-network-record",
        environment={"ALCHEMI_MASTER_ADDR": "10.117.10.97"},
        ip_script="""#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  "-4 -o addr show scope global")
    echo "2: eno1np0 inet 10.117.10.97/19 scope global eno1np0"
    ;;
  "-4 -o addr show dev eno1np0 scope global")
    echo "2: eno1np0 inet 10.117.10.97/19 scope global eno1np0"
    ;;
  *)
    echo "unexpected ip arguments: $*" >&2
    exit 2
    ;;
esac
""",
    )

    assert "address=10.117.10.97" in result.stdout
    assert "interface=eno1np0" in result.stdout
    assert "nccl==eno1np0" in result.stdout
    assert "gloo=eno1np0" in result.stdout
    assert "selection=automatic" in result.stdout


def test_domain_launcher_selects_the_worker_route_to_the_master(
    tmp_path: Path,
) -> None:
    result = _run_launcher_network_mode(
        tmp_path,
        mode="--print-network-record",
        environment={"ALCHEMI_MASTER_ADDR": "10.117.10.97"},
        ip_script="""#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  "-4 -o addr show scope global")
    echo "2: enp1s0f0np0 inet 10.176.188.32/16 scope global enp1s0f0np0"
    ;;
  "-4 route get 10.117.10.97")
    echo "10.117.10.97 via 10.176.0.1 dev enp1s0f0np0 src 10.176.188.32"
    ;;
  "-4 -o addr show dev enp1s0f0np0 scope global")
    echo "2: enp1s0f0np0 inet 10.176.188.32/16 scope global enp1s0f0np0"
    ;;
  *)
    echo "unexpected ip arguments: $*" >&2
    exit 2
    ;;
esac
""",
    )

    assert "address=10.176.188.32" in result.stdout
    assert "interface=enp1s0f0np0" in result.stdout
    assert "nccl==enp1s0f0np0" in result.stdout
    assert "gloo=enp1s0f0np0" in result.stdout
    assert "selection=automatic" in result.stdout


def test_domain_launcher_requires_an_explicit_interface_without_fallback(
    tmp_path: Path,
) -> None:
    valid = _run_launcher_network_mode(
        tmp_path / "valid",
        mode="--print-network-record",
        environment={
            "ALCHEMI_MASTER_ADDR": "10.117.10.97",
            "ALCHEMI_DISTRIBUTED_IFACE": "custom0",
        },
        ip_script="""#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  "-4 -o addr show scope global")
    echo "2: custom0 inet 10.176.188.32/16 scope global custom0"
    ;;
  "link show dev custom0")
    exit 0
    ;;
  "-4 route get 10.117.10.97 oif custom0")
    echo "10.117.10.97 via 10.176.0.1 dev custom0 src 10.176.188.32"
    ;;
  "-4 -o addr show dev custom0 scope global")
    echo "2: custom0 inet 10.176.188.32/16 scope global custom0"
    ;;
  *)
    echo "unexpected ip arguments: $*" >&2
    exit 2
    ;;
esac
""",
    )
    assert "interface=custom0" in valid.stdout
    assert "selection=explicit" in valid.stdout

    master = _run_launcher_network_mode(
        tmp_path / "master",
        mode="--print-network-record",
        environment={
            "ALCHEMI_MASTER_ADDR": "10.176.188.32",
            "ALCHEMI_DISTRIBUTED_IFACE": "custom0",
        },
        ip_script="""#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  "-4 -o addr show scope global"|"-4 -o addr show dev custom0 scope global")
    echo "2: custom0 inet 10.176.188.32/16 scope global custom0"
    ;;
  "link show dev custom0")
    exit 0
    ;;
  *)
    echo "unexpected ip arguments: $*" >&2
    exit 2
    ;;
esac
""",
    )
    assert "address=10.176.188.32" in master.stdout
    assert "interface=custom0" in master.stdout
    assert "selection=explicit" in master.stdout

    invalid = _run_launcher_network_mode(
        tmp_path / "invalid",
        mode="--print-network-record",
        environment={
            "ALCHEMI_MASTER_ADDR": "10.117.10.97",
            "ALCHEMI_DISTRIBUTED_IFACE": "missing0",
        },
        check=False,
        ip_script="""#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "-4 -o addr show scope global" ]]; then
  echo "2: test0 inet 10.176.188.32/16 scope global test0"
  exit 0
fi
if [[ "$*" == "link show dev missing0" ]]; then
  echo "missing interface" >&2
  exit 1
fi
echo "automatic fallback was attempted" >&2
exit 2
""",
    )
    assert invalid.returncode != 0
    assert "missing interface" in invalid.stderr
    assert "automatic fallback was attempted" not in invalid.stderr


@pytest.mark.parametrize(
    "mode_environment",
    (
        {"ALCHEMI_DOMAIN_GPUS": "1"},
        {
            "SLURM_JOB_ID": "123",
            "SLURM_NNODES": "2",
            "SLURM_NODEID": "1",
            "ALCHEMI_MASTER_ADDR": "127.0.0.1",
            "ALCHEMI_MASTER_PORT": "22123",
        },
    ),
    ids=("local", "slurm"),
)
def test_fresh_torchrun_imports_leave_staged_sources_bytecode_free(
    tmp_path: Path,
    mode_environment: dict[str, str],
) -> None:
    main_environment = tmp_path / "main-environment"
    python_overlay = tmp_path / "python-overlay"
    core_root = tmp_path / "toolkit-core"
    ops_root = tmp_path / "toolkit-ops"
    tutorial_root = tmp_path / "tutorial"
    for root, package_name in (
        (core_root, "core_probe"),
        (ops_root, "ops_probe"),
        (tutorial_root, "tutorial_probe"),
    ):
        package = root / package_name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            f'SOURCE = "{package_name}"\n',
            encoding="utf-8",
        )

    torchrun = python_overlay / "bin" / "torchrun"
    torchrun.parent.mkdir(parents=True)
    torchrun.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --standalone)
      shift
      ;;
    --nnodes|--nproc-per-node|--node-rank|--master-addr|--master-port)
      shift 2
      ;;
    *)
      break
      ;;
  esac
done
exec "$TEST_PYTHON" "$@"
""",
        encoding="utf-8",
    )
    torchrun.chmod(0o755)
    (python_overlay / ".part1-ready.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    ip = fake_bin / "ip"
    ip.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  "-4 -o addr show scope global"|"-4 -o addr show dev test0 scope global")
    echo "2: test0 inet 127.0.0.1/8 scope global test0"
    ;;
  *)
    echo "unexpected ip arguments: $*" >&2
    exit 2
    ;;
esac
""",
        encoding="utf-8",
    )
    ip.chmod(0o755)

    runner = tutorial_root / "runner.py"
    runner.write_text(
        """import os
import sys

import core_probe
import ops_probe
import tutorial_probe

assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
assert sys.dont_write_bytecode
assert core_probe.SOURCE == "core_probe"
assert ops_probe.SOURCE == "ops_probe"
assert tutorial_probe.SOURCE == "tutorial_probe"
if "SLURM_JOB_ID" in os.environ:
    assert os.environ["NCCL_SOCKET_IFNAME"] == "=test0"
    assert os.environ["GLOO_SOCKET_IFNAME"] == "test0"
""",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    for name in (
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPYCACHEPREFIX",
        "SLURM_JOB_ID",
        "SLURM_NNODES",
        "SLURM_NODEID",
        "ALCHEMI_MASTER_ADDR",
        "ALCHEMI_MASTER_PORT",
        "ALCHEMI_DOMAIN_GPUS",
        "ALCHEMI_PYTHON_OVERLAY",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "ALCHEMI_MAIN_ENV": str(main_environment),
            "ALCHEMI_PYTHON_OVERLAY": str(python_overlay),
            "ALCHEMI_TOOLKIT_CORE_ROOT": str(core_root),
            "ALCHEMI_TOOLKIT_OPS_ROOT": str(ops_root),
            "TEST_PYTHON": sys.executable,
            "PATH": f"{fake_bin}:{environment['PATH']}",
            **mode_environment,
        }
    )

    subprocess.run(
        ["bash", str(DOMAIN_LAUNCHER_PATH), str(runner)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert not list(tmp_path.rglob("__pycache__"))
    assert not list(tmp_path.rglob("*.pyc"))


def test_h100_notebook_run_writes_timings_before_validation() -> None:
    source = REMASTER_SBATCH_PATH.read_text(encoding="utf-8")

    assert 'TIMING_PATH="$FINAL_DIR/notebook-timings.json"' in source
    timing_argument = source.index('--timing-output "$TIMING_PATH"')
    validation = source.index('"$LOCAL_REPO/scripts/validate_part1_ir_run.py"')
    assert timing_argument < validation


@pytest.mark.parametrize("name", SBATCH_NAMES)
def test_h100_jobs_require_one_full_tutorial_commit(name: str) -> None:
    source = (REPOSITORY_ROOT / "scripts" / name).read_text(encoding="utf-8")

    assert "ALCHEMI_TUTORIAL_COMMIT" in source
    assert "^[0-9a-f]{40}$" in source
    assert 'rev-parse HEAD)" = "$TUTORIAL_COMMIT"' in source
    assert "status --porcelain=v1 --untracked-files=all" in source


def test_h100_notebook_uses_a_commit_only_local_checkout() -> None:
    source = REMASTER_SBATCH_PATH.read_text(encoding="utf-8")

    assert 'rsync -a "$SHARED_REPO/" "$LOCAL_REPO/"' not in source
    clone = source.index(
        'git clone --no-checkout --no-hardlinks "$SHARED_REPO" "$LOCAL_REPO"'
    )
    checkout = source.index('git -C "$LOCAL_REPO" checkout --detach "$TUTORIAL_COMMIT"')
    runtime_check = source.index(
        '"$PYTHON" "$LOCAL_REPO/build/verify_part1_runtime.py"'
    )
    d3_prewarm = source.index('"$PYTHON" "$LOCAL_REPO/build/prewarm_d3.py"')
    assert clone < checkout < runtime_check < d3_prewarm


def test_compute_lab_runbook_stages_both_toolkit_source_checkouts() -> None:
    source = COMPUTE_LAB_RUNBOOK_PATH.read_text(encoding="utf-8")

    expected = (
        (
            "ALCHEMI_TOOLKIT_CORE_ROOT",
            "https://github.com/NVIDIA/nvalchemi-toolkit.git",
            RUNTIME_CHECK.EXPECTED_CORE_COMMIT,
        ),
        (
            "ALCHEMI_TOOLKIT_OPS_ROOT",
            "https://github.com/NVIDIA/nvalchemi-toolkit-ops.git",
            RUNTIME_CHECK.EXPECTED_OPS_COMMIT,
        ),
    )
    for root_name, repository_url, commit in expected:
        assert f"export {root_name}=" in source
        assert repository_url in source
        assert commit in source
        assert f'git -C "${root_name}" checkout --detach' in source
        assert f'git -C "${root_name}" rev-parse HEAD' in source
        assert (
            f'git -C "${root_name}" \\\n'
            "    ls-files --others --ignored --exclude-standard"
        ) in source


def test_domain_job_fallbacks_match_the_documented_toolkit_checkout_paths() -> None:
    source = DOMAIN_SBATCH_PATH.read_text(encoding="utf-8")

    assert (
        'CORE_ROOT="${ALCHEMI_TOOLKIT_CORE_ROOT:-'
        '$RUN_ROOT/stage/toolkit/nvalchemi-toolkit-$CORE_COMMIT}"'
    ) in source
    assert (
        'OPS_ROOT="${ALCHEMI_TOOLKIT_OPS_ROOT:-'
        '$RUN_ROOT/stage/toolkit/nvalchemi-toolkit-ops-$OPS_COMMIT}"'
    ) in source
    assert "$RUN_ROOT/stage/nvalchemi-toolkit-0.2.0-rc" not in source
    assert "$RUN_ROOT/stage/nvalchemi-toolkit-ops-0.4.0" not in source


def test_domain_source_is_checked_before_repository_python_runs() -> None:
    source = DOMAIN_SBATCH_PATH.read_text(encoding="utf-8")

    source_check = source.index(
        'test "$(git -C "$SHARED_REPO" rev-parse HEAD)" = "$TUTORIAL_COMMIT"'
    )
    methodology_read = source.index("mapfile -t DOMAIN_METHODOLOGY_DEFAULTS")
    runtime_check = source.index(
        '"$PYTHON" "$SHARED_REPO/build/verify_part1_runtime.py"'
    )
    d3_prewarm = source.index('"$PYTHON" "$SHARED_REPO/build/prewarm_d3.py"')
    assert source_check < methodology_read < runtime_check < d3_prewarm


def test_domain_job_locks_and_rechecks_its_reportable_sources() -> None:
    source = DOMAIN_SBATCH_PATH.read_text(encoding="utf-8")

    runtime_check = source.index(
        '"$PYTHON" "$SHARED_REPO/build/verify_part1_runtime.py"'
    )
    assert "--require-clean-source" in source[runtime_check:]
    producer_write = source.index('> "$FINAL_DIR/producer-SHA256SUMS"')
    first_campaign_command = source.index(
        '"$PYTHON" "$PLAN_SCRIPT" checkpoint-preflight'
    )
    producer_recheck = source.rindex('sha256sum -c "$FINAL_DIR/producer-SHA256SUMS"')
    artifact_index = source.index('> "$FINAL_DIR/artifact-SHA256SUMS"')
    assert producer_write < first_campaign_command < producer_recheck < artifact_index
    assert (
        source.count(
            'test "$(git -C "$SHARED_REPO" rev-parse HEAD)" = "$TUTORIAL_COMMIT"'
        )
        == 2
    )
    assert (
        source.count(
            'git -C "$SHARED_REPO" ls-files --others --ignored --exclude-standard'
        )
        == 2
    )
    for root_name in ("CORE_ROOT", "OPS_ROOT"):
        assert (
            f'git -C "${root_name}" ls-files --others --ignored --exclude-standard'
        ) in source
    assert "--output-dir $RUN_ROOT/results/domain-fixed-bundle" in source
    assert '--base-box-dir "$BASE_BOX_DIR"' in source
    assert 'command -v "$PACKMOL"' not in source
    producer_command = source[
        source.rfind("sha256sum", 0, producer_write) : producer_write
    ]
    for base_file in (
        '"$BASE_BOX_MANIFEST"',
        '"$BASE_BOX_STRUCTURE"',
        '"$BASE_BOX_CHECKSUMS"',
    ):
        assert base_file in producer_command
    assert (
        "--output-dir "
        "$SHARED_REPO/part-1-scalable-atomistic-workflows/"
        "data/domain_decomposition/recorded"
    ) not in source


@pytest.mark.parametrize(
    ("path", "first_execution"),
    (
        (DOMAIN_SBATCH_PATH, "mapfile -t DOMAIN_METHODOLOGY_DEFAULTS"),
        (REMASTER_SBATCH_PATH, '"$PYTHON" -m ipykernel install'),
        (DOMAIN_LAUNCHER_PATH, 'exec "$TORCHRUN"'),
    ),
)
def test_measured_jobs_require_the_verified_python_overlay(
    path: Path,
    first_execution: str,
) -> None:
    source = path.read_text(encoding="utf-8")

    for check in (': "${ALCHEMI_PYTHON_OVERLAY:?', 'test -s "$READY_FILE"'):
        assert check in source
    ready_assignment = (
        'READY_FILE="$ALCHEMI_PYTHON_OVERLAY/.part1-ready.json"'
        if path == DOMAIN_LAUNCHER_PATH
        else 'READY_FILE="$PYTHON_OVERLAY/.part1-ready.json"'
    )
    assert ready_assignment in source
    assert 'PYTHON_OVERLAY="${ALCHEMI_PYTHON_OVERLAY:-}"' not in source
    assert source.index('test -s "$READY_FILE"') < source.index(first_execution)


def test_h100_jobs_use_overlay_executables_and_the_conda_base_path() -> None:
    setup_source = SETUP_SBATCH_PATH.read_text(encoding="utf-8")
    domain_source = DOMAIN_SBATCH_PATH.read_text(encoding="utf-8")
    notebook_source = REMASTER_SBATCH_PATH.read_text(encoding="utf-8")
    launcher_source = DOMAIN_LAUNCHER_PATH.read_text(encoding="utf-8")

    for source in (setup_source, domain_source, notebook_source):
        assert 'PYTHON="$PYTHON_OVERLAY/bin/python"' in source
        assert 'export PATH="$MAIN_ENV/bin:$PATH"' in source
        assert 'PYTHON="$MAIN_ENV/bin/python"' not in source
        assert (
            'export PYTHONPATH="$OVERLAY_SITE${PYTHONPATH:+:$PYTHONPATH}"' not in source
        )
    assert 'TORCHRUN="$ALCHEMI_PYTHON_OVERLAY/bin/torchrun"' in launcher_source
    assert 'TORCHRUN="$ALCHEMI_MAIN_ENV/bin/torchrun"' not in launcher_source
    assert 'JUPYTER="$PYTHON_OVERLAY/bin/jupyter"' in notebook_source
    assert '"$JUPYTER" kernelspec list' in notebook_source


def test_setup_creates_or_strictly_reuses_an_immutable_conda_base() -> None:
    source = SETUP_SBATCH_PATH.read_text(encoding="utf-8")

    for check in (
        ': "${ALCHEMI_CONDA_EXE:?',
        'CONDA_EXE="$ALCHEMI_CONDA_EXE"',
        'MAIN_ENV_REQUESTED="$ALCHEMI_MAIN_ENV"',
        'PYTHON_OVERLAY_REQUESTED="$ALCHEMI_PYTHON_OVERLAY"',
        'MAIN_ENV="$(realpath -m "$MAIN_ENV_REQUESTED")"',
        'PYTHON_OVERLAY="$(realpath -m "$PYTHON_OVERLAY_REQUESTED")"',
        '[[ -L "$MAIN_ENV_REQUESTED" && ! -e "$MAIN_ENV_REQUESTED" ]]',
        '[[ "$PYTHON_OVERLAY/" == "$MAIN_ENV/"* ]]',
        'if [[ -e "$MAIN_ENV" || -L "$MAIN_ENV" ]]; then',
        'echo "Refusing existing partial or invalid Conda base: $MAIN_ENV"',
        '"$CONDA_EXE" env create',
        '--prefix "$MAIN_ENV"',
        '--file "$ENVIRONMENT_SPEC"',
        'test -d "$MAIN_ENV/conda-meta"',
        'test -x "$MAIN_ENV/bin/python"',
        'test -x "$MAIN_ENV/bin/uv"',
        'test -x "$MAIN_ENV/bin/packmol"',
        'test -x "$MAIN_ENV/bin/dot"',
        'BASE_SPEC_MARKER="$MAIN_ENV/.part1-environment.sha256"',
        '"$(< "$BASE_SPEC_MARKER")" != "$ENVIRONMENT_SPEC_SHA256"',
        '"$CONDA_EXE" list --prefix "$MAIN_ENV" --explicit',
        '"ovito": "3.15.4"',
        '"packmol": "21.2.1"',
        '"python": "3.12.13"',
        "uv_version < (0, 9, 26)",
        'if "graphviz" not in records',
    ):
        assert check in source
    assert 'ENVIRONMENT_SPEC="$SHARED_REPO/build/environment.yml"' in source
    assert " conda install " not in source
    assert " env update " not in source

    overlay_refusal = source.index(
        'echo "Refusing to replace existing package layer: $PYTHON_OVERLAY_REQUESTED"'
    )
    base_creation = source.index('"$CONDA_EXE" env create')
    overlay_creation = source.index('"$MAIN_ENV/bin/uv" venv')
    assert overlay_refusal < base_creation < overlay_creation


def test_setup_installs_the_complete_requirements_into_a_system_site_overlay() -> None:
    source = SETUP_SBATCH_PATH.read_text(encoding="utf-8")

    venv_start = source.index('"$MAIN_ENV/bin/uv" venv')
    install_start = source.index('"$MAIN_ENV/bin/uv" pip install')
    install_end = source.index('"$MAIN_ENV/bin/uv" pip check')
    venv_block = source[venv_start:install_start]
    install_block = source[install_start:install_end]
    assert "--system-site-packages" in venv_block
    assert '--python "$MAIN_ENV/bin/python"' in venv_block
    for check in (
        '--python "$PYTHON_OVERLAY/bin/python"',
        "--no-sources-package nvalchemi-toolkit-ops",
        "--torch-backend cu130",
        '--requirements "$REQUIREMENTS"',
    ):
        assert check in install_block
    assert 'REQUIREMENTS="$SHARED_REPO/build/requirements.txt"' in source
    uv_record = source.index(
        '"$MAIN_ENV/bin/uv" --version | tee "$RESULT_DIR/uv-version.txt"'
    )
    assert uv_record < install_start
    assert '"$RESULT_DIR/uv-version.txt"' in source


def test_setup_prewarms_model_and_d3_files_before_marking_ready() -> None:
    source = (
        REPOSITORY_ROOT / "scripts" / "slurm_part1_sevennet_setup.sbatch"
    ).read_text(encoding="utf-8")

    ready_write = source.index('mv "$READY_TMP" "$READY_FILE"')
    for command in (
        '"$SHARED_REPO/build/prewarm_aimnet.py"',
        '"$SHARED_REPO/build/prewarm_d3.py"',
        '"$SHARED_REPO/build/prewarm_sevennet.py"',
    ):
        assert source.index(command) < ready_write
    assert "--device cuda" in source
    assert '--parameter-file "$ALCHEMI_D3_PARAM_FILE"' in source
    assert '"$RESULT_DIR/aimnet-cache.json"' in source
    assert '"$RESULT_DIR/d3-cache.json"' in source


def test_setup_records_the_qualified_current_release_test_set() -> None:
    source = SETUP_SBATCH_PATH.read_text(encoding="utf-8")
    runbook = COMPUTE_LAB_RUNBOOK_PATH.read_text(encoding="utf-8")

    pytest_run = source.index('"$PYTHON" -m pytest -q')
    ready_write = source.index('mv "$READY_TMP" "$READY_FILE"')
    assert pytest_run < ready_write
    pytest_block = source[pytest_run:ready_write]
    for test_root in ("tests", "reference/tests"):
        assert (
            f'"$SHARED_REPO/part-1-scalable-atomistic-workflows/{test_root}"'
        ) in pytest_block
    for name in RETIRED_ORBMOL_TESTS:
        assert (
            f'--ignore="$SHARED_REPO/part-1-scalable-atomistic-workflows/tests/{name}"'
        ) in pytest_block
    assert '| tee "$RESULT_DIR/pytest.txt"' in pytest_block
    assert '"$RESULT_DIR/pytest.txt"' in source
    assert "complete Part 1 test suite" not in source
    assert "complete Part 1 test suite" not in runbook
    assert "unqualified run of every retained test file" in runbook


def test_setup_routes_pytest_cache_to_the_result_directory() -> None:
    source = SETUP_SBATCH_PATH.read_text(encoding="utf-8")

    pytest_run = source.index('"$PYTHON" -m pytest -q')
    pytest_output = source.index('| tee "$RESULT_DIR/pytest.txt"', pytest_run)
    pytest_invocation = source[pytest_run:pytest_output]

    assert '-o "cache_dir=$RESULT_DIR/pytest-cache"' in pytest_invocation


def test_launch_path_contains_no_runtime_patch_or_base_python_fallback() -> None:
    paths = (
        SETUP_SBATCH_PATH,
        DOMAIN_SBATCH_PATH,
        REMASTER_SBATCH_PATH,
        DOMAIN_LAUNCHER_PATH,
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for forbidden in ("scripts/patches/", "git apply", "patch -p", "sed -i"):
            assert forbidden not in source


def test_notebook_requires_domain_results_and_ignores_pipeline_results() -> None:
    source = REMASTER_SBATCH_PATH.read_text(encoding="utf-8")

    domain_check = source.index('cd "$PART_REL/data/domain_decomposition/recorded"')
    assert 'test -d "$PART_REL/data/domain_decomposition/recorded"' in source
    assert domain_check < source.index(
        '"$PYTHON" "$LOCAL_REPO/build/verify_part1_runtime.py"'
    )
    assert "compute_lab_pipeline_campaign" not in source
    assert "DistributedPipeline campaign bundle" not in source


def test_runbook_checks_domain_jobs_and_scopes_the_signed_result_commit() -> None:
    source = COMPUTE_LAB_RUNBOOK_PATH.read_text(encoding="utf-8")

    for check in (
        'DOMAIN_JOB_IDS="${domain_jobs[1]},${domain_jobs[2]},${domain_jobs[4]}"',
        "sacct -X",
        ')" -eq 3',
        '$3 != "COMPLETED"',
        '$4 != "0:0"',
        'DOMAIN_BUNDLE_NAME="REPLACE_WITH_DOMAIN_FIXED_BUNDLE_NAME"',
        'REMOTE_DOMAIN_BUNDLE="$COMPUTE_LAB_RUN_ROOT/results/$DOMAIN_BUNDLE_NAME"',
        'TRANSFER_ROOT="$(',
        'mktemp -d "${TMPDIR:-/tmp}/alchemi-domain-bundle.XXXXXX"',
        "ssh cl",
        'scp -pr "cl:$REMOTE_DOMAIN_BUNDLE/." "$TRANSFER_BUNDLE/"',
        'cd "$TRANSFER_BUNDLE"',
        'RECORDED_REL="part-1-scalable-atomistic-workflows/'
        'data/domain_decomposition/recorded"',
        'cp -a "$TRANSFER_BUNDLE/." "$RECORDED_DIR/"',
        'git -C "$DEVELOPMENT_REPO" add -- "$RECORDED_REL"',
        'git -C "$DEVELOPMENT_REPO" commit -s',
        'R2="$(git -C "$DEVELOPMENT_REPO" rev-parse HEAD)"',
        '[[ "$R2" =~ ^[0-9a-f]{40}$ ]]',
        "Stop for user review and approval before staging, committing, or pushing",
    ):
        assert check in source


def test_reportable_domain_run_uses_only_the_versioned_fixed_work() -> None:
    source = DOMAIN_SBATCH_PATH.read_text(encoding="utf-8")

    for check in (
        "config.fixed_molecules_per_species",
        "config.electrostatics_validation_molecules_per_species",
        'print(" ".join(map(str, config.campaign_world_sizes)))',
        "config.evaluation_warmup_count",
        "config.evaluation_pass_count",
        'FIXED_PAIRS="${DOMAIN_DEFAULTS[0]}"',
        'if [[ " $SUPPORTED_WORLD_SIZES " != *" $WORLD_SIZE "* ]]',
        "This example supports --nodes 1, 2, or 4",
        "The recorded example requires one warmup and three measured passes",
    ):
        assert check in source
    for removed in (
        "ALCHEMI_DOMAIN_PAIR_COUNTS",
        "ALCHEMI_DOMAIN_PHASE",
        "ALCHEMI_DOMAIN_CAPACITY_DIR",
        "PARITY_PAIRS",
    ):
        assert removed not in source


def test_domain_launch_resolves_and_records_each_rank_interface() -> None:
    source = DOMAIN_SBATCH_PATH.read_text(encoding="utf-8")
    launcher = DOMAIN_LAUNCHER_PATH.read_text(encoding="utf-8")
    runbook = COMPUTE_LAB_RUNBOOK_PATH.read_text(encoding="utf-8")
    runbook_text = " ".join(runbook.split())

    expected = (
        'MASTER_NODE="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"',
        "export MASTER_NODE",
        "export ALCHEMI_MASTER_ADDR",
        'ALCHEMI_MASTER_ADDR="$(',
        'srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="$MASTER_NODE"',
        '"$LAUNCHER" --print-master-address',
        'if [[ -z "$ALCHEMI_MASTER_ADDR" ]]',
        "Could not resolve a routed global IPv4 address on $MASTER_NODE",
        '"$LAUNCHER" --print-network-record',
        "| LC_ALL=C sort",
        '> "$FINAL_DIR/network-interfaces.txt"',
        '"$FINAL_DIR/network-interfaces.txt"',
    )
    for check in expected:
        assert check in source

    master_node = source.index('MASTER_NODE="$(scontrol show hostnames')
    master_resolution = source.index('ALCHEMI_MASTER_ADDR="$(')
    master_validation = source.index('if [[ -z "$ALCHEMI_MASTER_ADDR" ]]')
    network_record = source.index('> "$FINAL_DIR/network-interfaces.txt"')
    launcher_call = source.index('"$LAUNCHER" "$RUNNER"')
    artifact_record = source.index(
        '"$FINAL_DIR/network-interfaces.txt"',
        network_record + len('> "$FINAL_DIR/network-interfaces.txt"'),
    )
    assert (
        master_node
        < master_resolution
        < master_validation
        < network_record
        < launcher_call
        < artifact_record
    )
    assert (
        'ALCHEMI_MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST"' not in source
    )
    assert "enp1s0f0np0" not in source
    assert "eno1np0" not in source
    assert "169.254" not in source
    assert "10[.]63" not in source
    assert "10.63" not in source
    assert "NCCL_IB_DISABLE" not in source

    interface_resolution = launcher.index("resolve_local_network")
    nccl_export = launcher.index('export NCCL_SOCKET_IFNAME="=$interface"')
    gloo_export = launcher.index('export GLOO_SOCKET_IFNAME="$interface"')
    torchrun = launcher.index('exec "$TORCHRUN"')
    assert interface_resolution < nccl_export < gloo_export < torchrun
    assert "--print-master-address" in launcher
    assert "--print-network-record" in launcher
    assert "enp1s0f0np0" not in launcher
    assert "eno1np0" not in launcher
    assert "NCCL_IB_DISABLE" not in launcher
    assert "does not prove that NCCL" in runbook_text
    assert "Use one checked node family for the recorded 1/2/4-GPU set" in runbook_text
    assert "Test a mixed allocation with a real NCCL collective" in runbook_text


@pytest.mark.parametrize(
    "path",
    (DOMAIN_SBATCH_PATH, DOMAIN_LAUNCHER_PATH),
    ids=("sbatch", "launcher"),
)
def test_domain_shell_launchers_have_valid_bash_syntax(path: Path) -> None:
    subprocess.run(
        ["bash", "-n", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_successful_domain_launch_without_result_records_exit_code_66() -> None:
    source = DOMAIN_SBATCH_PATH.read_text(encoding="utf-8")
    run_case_start = source.index("run_case() {")
    run_case_end = source.index(
        '\n}\n\nif [[ "$WORLD_SIZE" -eq 1 ]]',
        run_case_start,
    )
    run_case = source[run_case_start:run_case_end]

    missing_result = run_case.index('elif [[ ! -s "$CURRENT_OUTPUT" ]]')
    diagnostic = run_case.index(
        "The launcher exited successfully but wrote no result JSON.",
        missing_result,
    )
    recorded_failure = run_case.index("record_failure 66", diagnostic)
    assert missing_result < diagnostic < recorded_failure


def test_domain_commands_pass_the_input_path_once_each() -> None:
    source = DOMAIN_SBATCH_PATH.read_text(encoding="utf-8")
    record_start = source.index("record_failure() {")
    record_end = source.index(
        "\n}\n\nrecord_active_timeout()",
        record_start,
    )
    run_start = source.index("run_case() {")
    run_end = source.index(
        '\n}\n\nif [[ "$WORLD_SIZE" -eq 1 ]]',
        run_start,
    )

    assert source[record_start:record_end].count('--input-extxyz "$CURRENT_INPUT"') == 1
    assert source[run_start:run_end].count('--input-extxyz "$CURRENT_INPUT"') == 1


def test_domain_hardware_counts_come_from_observed_runtime_rows() -> None:
    source = DOMAIN_PLAN_PATH.read_text(encoding="utf-8")

    for check in (
        "max_observed_gpus = max(",
        "max_observed_nodes = max(",
        'str(runtime["host"])',
        '"gpus_available": max_observed_gpus',
        '"nodes_available": max_observed_nodes',
    ):
        assert check in source


def test_domain_hardware_identity_records_source_fields() -> None:
    plan_source = DOMAIN_PLAN_PATH.read_text(encoding="utf-8")
    results_source = DOMAIN_RESULTS_PATH.read_text(encoding="utf-8")

    expected_plan_values = {
        "site_source": "operator-declared",
        "resource_count_source": ("derived from successful per-rank runtime records"),
        "interconnect_source": ("operator-declared; raw GPU topology is retained"),
    }
    for field, value in expected_plan_values.items():
        assert f'"{field}":' in plan_source
        assert value in plan_source
        assert f'"{field}",' in results_source


def test_full_notebook_job_keeps_checksum_failures() -> None:
    source = (
        REPOSITORY_ROOT / "scripts" / "slurm_part1_remaster_h100.sbatch"
    ).read_text(encoding="utf-8")

    assert "--require-clean-source" in source
    assert "checksum_status=$?" in source
    assert 'if [[ "$status" -eq 0 && "$checksum_status" -ne 0 ]]' in source
    assert 'status="$checksum_status"' in source
