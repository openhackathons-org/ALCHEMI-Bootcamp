"""Static checks for the dependencies installed in the Part 1 image."""

from __future__ import annotations

import importlib.util
from pathlib import Path, PurePosixPath
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]

RUNBOOK_SCRIPT_FILES = {
    "scripts/assemble_part1_pipeline_campaign.py",
    "scripts/benchmark_part1_distributed_campaign.py",
    "scripts/part1_domain_plan.py",
    "scripts/review_part1_ir_executed_notebook.py",
    "scripts/run_part1_domain_decomposition.sh",
    "scripts/slurm_part1_distributed_campaign.sbatch",
    "scripts/slurm_part1_domain_decomposition.sbatch",
    "scripts/slurm_part1_remaster_h100.sbatch",
    "scripts/slurm_part1_sevennet_setup.sbatch",
    "scripts/validate_part1_ir_run.py",
}

PACKAGED_IMAGE_TEST_FILES = {
    "part-1-scalable-atomistic-workflows/tests/test_artifact_manifest.py",
    "part-1-scalable-atomistic-workflows/tests/test_experimental_reference.py",
    "part-1-scalable-atomistic-workflows/tests/test_notebook_contract.py",
    "part-1-scalable-atomistic-workflows/tests/test_reference_data.py",
}

RETIRED_ORBMOL_TEST_FILES = {
    "part-1-scalable-atomistic-workflows/tests/test_orbmol_adapter.py",
    "part-1-scalable-atomistic-workflows/tests/test_orbmol_checks.py",
    "part-1-scalable-atomistic-workflows/tests/test_orbmol_config.py",
}

PART1_PREFIX = "part-1-scalable-atomistic-workflows/"

EXPECTED_PART1_EXCLUSION_RULES = {
    f"{PART1_PREFIX}outputs/",
    f"{PART1_PREFIX}aux/benchmark_results.py",
    f"{PART1_PREFIX}aux/models/orbmol.py",
    f"{PART1_PREFIX}aux/models/orbmol_checks.py",
    f"{PART1_PREFIX}aux/models/orbmol_config.py",
    f"{PART1_PREFIX}aux/models/orbmol_wrapper.py",
    f"{PART1_PREFIX}data/compute_lab_distributed_pipeline/",
    f"{PART1_PREFIX}data/compute_lab_distributed_pipeline.provenance.md",
    f"{PART1_PREFIX}assets/images/banner_candidates/README.md",
    f"{PART1_PREFIX}assets/images/banner_candidates/water-ir-v2-01-hydrogen-bond-landscape.png",
    f"{PART1_PREFIX}assets/images/banner_candidates/water-ir-v2-02-batched-water-systems.png",
    f"{PART1_PREFIX}assets/images/banner_candidates/water-ir-v2-03-composed-potential.png",
    f"{PART1_PREFIX}tests/*",
    f"{PART1_PREFIX}reference/tests/test_b97_3c_ir.py",
    f"{PART1_PREFIX}reference/artifacts/water_dimer_b97_3c/timer.dat",
    f"{PART1_PREFIX}reference/artifacts/provenance/**/*.npy",
    f"{PART1_PREFIX}reference/artifacts/provenance/**/*.npz",
    f"{PART1_PREFIX}reference/artifacts/provenance/**/*.csv",
    f"{PART1_PREFIX}reference/artifacts/provenance/**/*.xyz",
    f"{PART1_PREFIX}reference/artifacts/provenance/**/*.out",
    f"{PART1_PREFIX}reference/artifacts/provenance/**/*.dat",
    f"{PART1_PREFIX}reference/artifacts/provenance/**/*.txt",
    f"{PART1_PREFIX}reference/artifacts/provenance/**/diagnostics.json",
    f"{PART1_PREFIX}reference/artifacts/provenance/**/run_config.json",
    f"{PART1_PREFIX}reference/artifacts/provenance/h6/artifacts/",
}


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _plain(text: str) -> str:
    without_comment_markers = re.sub(r"(?m)^\s*#\s?", "", text)
    return re.sub(r"\s+", " ", without_comment_markers)


def _load_runtime_check():
    path = ROOT / "build" / "verify_part1_runtime.py"
    spec = importlib.util.spec_from_file_location("part1_runtime_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _requirement_names(requirements: str) -> set[str]:
    names: set[str] = set()
    for raw_line in requirements.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"[A-Za-z0-9_.-]+", line)
        assert match is not None, f"could not read requirement line: {raw_line!r}"
        names.add(match.group(0).lower())
    return names


def _source_manifest_paths() -> tuple[str, ...]:
    paths: list[str] = []
    for raw_line in _read("build/part1-source-files.txt").splitlines():
        relative_path = raw_line.strip()
        if relative_path and not relative_path.startswith("#"):
            paths.append(relative_path)
    return tuple(paths)


def _normalize_link_path(source: str, target: str) -> PurePosixPath:
    parts: list[str] = []
    for part in (PurePosixPath(source).parent / target).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return PurePosixPath(*parts)


def _is_intentionally_excluded_part1_file(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    if any(
        part in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
        for part in path.parts
    ):
        return True
    if path.suffix in {".pyc", ".pyo"} or path.name == ".DS_Store":
        return True
    if relative_path.startswith(f"{PART1_PREFIX}outputs/"):
        return True
    if relative_path.startswith(f"{PART1_PREFIX}tests/"):
        return relative_path not in PACKAGED_IMAGE_TEST_FILES
    if relative_path.startswith(
        f"{PART1_PREFIX}data/compute_lab_distributed_pipeline/"
    ):
        return True
    if relative_path.startswith(
        f"{PART1_PREFIX}reference/artifacts/provenance/h6/artifacts/"
    ):
        return True
    provenance_prefix = f"{PART1_PREFIX}reference/artifacts/provenance/"
    if relative_path.startswith(provenance_prefix):
        return path.suffix in {
            ".npy",
            ".npz",
            ".csv",
            ".xyz",
            ".out",
            ".dat",
            ".txt",
        } or path.name in {"diagnostics.json", "run_config.json"}
    return relative_path in {
        rule for rule in EXPECTED_PART1_EXCLUSION_RULES if "*" not in rule
    }


def test_part1_requirements_exclude_legacy_orb_packages() -> None:
    names = _requirement_names(_read("build/requirements.txt"))

    assert "orb-models" not in names
    assert "loguru" not in names


def test_docker_build_checks_the_part1_environment_without_an_override() -> None:
    dockerfile = _read("build/Dockerfile")

    assert "--overrides" not in dockerfile
    assert "build/overrides.txt" not in dockerfile
    assert "orb_models" not in dockerfile
    assert "loguru" not in dockerfile
    assert "RUN uv pip check" in dockerfile
    assert dockerfile.index("-r /tmp/requirements.txt") < dockerfile.index(
        "RUN uv pip check"
    )
    assert dockerfile.index("RUN uv pip check") < dockerfile.index(
        "# Fail the build loudly if a required Part 1 import is broken."
    )
    assert "COPY build/verify_part1_runtime.py" in dockerfile
    assert "COPY build/part1-source-files.txt" in dockerfile
    assert "python build/verify_part1_runtime.py" in dockerfile
    assert "--skip-source-check" in dockerfile
    assert "--output build/part1-image-runtime.json" in dockerfile
    aimnet_check = dockerfile.index(
        "RUN python /tmp/prewarm_aimnet.py --remove-checkpoints-after-test"
    )
    d3_check = dockerfile.index("RUN python /tmp/prewarm_d3.py")
    d3_remove = dockerfile.index(
        "rm -f /tmp/part1-dftd3-parameters.pt /tmp/part1-d3-check.json"
    )
    final_content = dockerfile.index("# Bake tutorial content into the image")
    assert aimnet_check < final_content
    assert d3_check < d3_remove < final_content
    runtime_check = dockerfile.index("python build/verify_part1_runtime.py")
    notebook_check = dockerfile.index(
        "part-1-scalable-atomistic-workflows/tests/test_notebook_contract.py"
    )
    assert final_content < runtime_check < notebook_check
    for test_name in (
        "test_artifact_manifest.py",
        "test_reference_data.py",
        "test_experimental_reference.py",
    ):
        assert test_name in dockerfile[notebook_check:]


def test_clean_image_contains_every_script_named_by_the_compute_runbook() -> None:
    dockerfile = _read("build/Dockerfile")
    runbook = _read("part-1-scalable-atomistic-workflows/COMPUTE_LAB_RUNBOOK.md")
    source_manifest = _read("build/part1-source-files.txt")

    observed = set(
        re.findall(r"scripts/[A-Za-z0-9_.-]+\.(?:py|sh|sbatch)", runbook)
    )
    assert observed == RUNBOOK_SCRIPT_FILES
    for relative_path in sorted(RUNBOOK_SCRIPT_FILES):
        assert relative_path in dockerfile
        assert relative_path in source_manifest


def test_clean_image_contains_every_declared_part1_source_file() -> None:
    dockerfile = _read("build/Dockerfile")
    source_paths = _source_manifest_paths()

    assert len(source_paths) == len(set(source_paths))
    assert {
        ".dockerignore",
        ".gitignore",
        "build/Dockerfile",
        "build/docker-compose.yml",
        "build/environment.yml",
    }.issubset(source_paths)
    for relative_path in source_paths:
        assert (ROOT / relative_path).is_file(), relative_path
        if relative_path.startswith("part-1-scalable-atomistic-workflows/"):
            assert (
                "COPY part-1-scalable-atomistic-workflows "
                "./part-1-scalable-atomistic-workflows"
            ) in dockerfile
        else:
            assert relative_path in dockerfile, relative_path


def test_clean_image_ships_only_the_tests_run_during_build() -> None:
    dockerfile = _read("build/Dockerfile")
    dockerignore_lines = {
        line.strip()
        for line in _read(".dockerignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    packaged_tests = set(
        re.findall(
            r"part-1-scalable-atomistic-workflows/tests/test_[A-Za-z0-9_]+\.py",
            dockerfile,
        )
    )

    assert packaged_tests == PACKAGED_IMAGE_TEST_FILES
    assert "part-1-scalable-atomistic-workflows/tests/*" in dockerignore_lines
    for relative_path in PACKAGED_IMAGE_TEST_FILES:
        assert f"!{relative_path}" in dockerignore_lines
    for relative_path in RETIRED_ORBMOL_TEST_FILES:
        assert relative_path not in packaged_tests
        assert f"!{relative_path}" not in dockerignore_lines


def test_part1_directory_copy_has_no_undeclared_files() -> None:
    source_paths = set(_source_manifest_paths())
    dockerignore_lines = {
        line.strip()
        for line in _read(".dockerignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert EXPECTED_PART1_EXCLUSION_RULES.issubset(dockerignore_lines)

    unclassified: list[str] = []
    part1_root = ROOT / PART1_PREFIX
    for path in sorted(item for item in part1_root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(ROOT).as_posix()
        if relative_path in source_paths or relative_path in PACKAGED_IMAGE_TEST_FILES:
            continue
        if not _is_intentionally_excluded_part1_file(relative_path):
            unclassified.append(relative_path)

    assert unclassified == []


def test_relative_links_resolve_within_the_clean_image() -> None:
    source_paths = set(_source_manifest_paths())
    image_paths = source_paths | PACKAGED_IMAGE_TEST_FILES
    image_directories = {
        parent
        for relative_path in image_paths
        for parent in PurePosixPath(relative_path).parents
    }
    link_pattern = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
    failures: list[str] = []

    for source in sorted(path for path in source_paths if path.endswith(".md")):
        for raw_target in link_pattern.findall(_read(source)):
            target = raw_target.strip().split("#", 1)[0]
            if (
                not target
                or "://" in target
                or target.startswith(("mailto:", "/"))
            ):
                continue
            resolved = _normalize_link_path(source, target)
            if str(resolved) not in image_paths and resolved not in image_directories:
                failures.append(f"{source}: {target} -> {resolved}")

    assert failures == []


def test_baked_runtime_check_can_run_without_git_metadata(tmp_path: Path) -> None:
    runtime_check = _load_runtime_check()

    report = runtime_check.source_report(
        tmp_path,
        require_clean_source=False,
        skip_source_check=True,
    )

    assert report["checked"] is False
    assert report["clean_checkout"] is None
    assert "only the baked runtime" in report["reason"]
    with pytest.raises(ValueError, match="mutually exclusive"):
        runtime_check.source_report(
            tmp_path,
            require_clean_source=True,
            skip_source_check=True,
        )


def test_part1_slurm_setup_checks_the_environment_without_an_override() -> None:
    setup = _read("scripts/slurm_part1_sevennet_setup.sbatch")
    nci_job = _read("scripts/slurm_part1_nci_stage3.sbatch")

    assert "--overrides" not in setup
    assert "build/overrides.txt" not in setup
    assert '"$MAIN_ENV/bin/uv" pip check' in setup
    assert setup.index('"$MAIN_ENV/bin/uv" pip install') < setup.index(
        '"$MAIN_ENV/bin/uv" pip check'
    )
    assert "build/overrides.txt" not in nci_job


def test_legacy_override_is_documented_but_not_part_of_part1_source() -> None:
    override_note = _read("build/overrides.txt")
    source_manifest = _read("build/part1-source-files.txt")

    assert "Historical record only." in override_note
    assert (
        "current Part 1 Docker image and Slurm setup do not read this file"
        in _plain(override_note)
    )
    assert "build/overrides.txt" not in source_manifest


def test_docs_put_retained_notebooks_in_separate_environments() -> None:
    readme = _read("README.md")
    notices = _read("THIRD_PARTY_NOTICES.md")
    snapshot = _read("RUNTIME_SNAPSHOT.md")
    part3_readme = _read("part-3-batched-melting-toolkit/README.md")

    plain_readme = _plain(readme)
    for term in (
        "separate historical MACE environment",
        "separate historical Orb environment",
        "does not install `orb-models` or the legacy-only `loguru` dependency",
    ):
        assert term in plain_readme

    assert "current Part 1 image does not install `orb-models`" in _plain(notices)
    assert (
        "Run the retained notebook in its separate historical environment."
        in _plain(notices)
    )
    assert "`build/Dockerfile` and `build/requirements.txt`" in snapshot
    assert "That file is retained only as a historical record" in snapshot
    assert "does not run in the remastered Part 1 image" in part3_readme
    assert "historical Orb environment separate" in _plain(part3_readme)
