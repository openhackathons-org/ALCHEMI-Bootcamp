"""Focused tests for synchronized runtime asset preparation and validation."""

from __future__ import annotations

import errno
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import NoReturn

import check_runtime
import prewarm_assets
import pytest


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unexpected_create(_path: Path) -> NoReturn:
    raise AssertionError("a verified cached asset must not be regenerated")


def _staging_entries(target: Path) -> list[Path]:
    if not target.parent.exists():
        return []
    return list(target.parent.glob(f".{target.stem}-*"))


def test_ensure_pinned_asset_creates_in_clean_custom_runtime(tmp_path: Path) -> None:
    payload = b"deterministic D3 parameter fixture"
    target = tmp_path / "custom-runtime" / "dftd3" / "dftd3_parameters.pt"
    created_paths: list[Path] = []

    def create(staged_path: Path) -> None:
        created_paths.append(staged_path)
        assert staged_path.name == target.name
        staged_path.write_bytes(payload)

    result = prewarm_assets.ensure_pinned_asset(
        target,
        expected_sha256=_sha256(payload),
        label="D3 parameters",
        create=create,
    )

    assert result == target.resolve()
    assert target.read_bytes() == payload
    assert len(created_paths) == 1
    assert created_paths[0].parent != target.parent
    assert _staging_entries(target) == []


def test_ensure_pinned_asset_reuses_verified_cache(tmp_path: Path) -> None:
    payload = b"already verified"
    target = tmp_path / "dftd3_parameters.pt"
    target.write_bytes(payload)

    result = prewarm_assets.ensure_pinned_asset(
        target,
        expected_sha256=_sha256(payload),
        label="D3 parameters",
        create=_unexpected_create,
    )

    assert result == target.resolve()
    assert target.read_bytes() == payload


def test_ensure_pinned_asset_preserves_mismatched_cache(tmp_path: Path) -> None:
    target = tmp_path / "dftd3_parameters.pt"
    target.write_bytes(b"unexpected existing data")

    with pytest.raises(RuntimeError, match="D3 parameters.*expected.*found"):
        prewarm_assets.ensure_pinned_asset(
            target,
            expected_sha256=_sha256(b"expected data"),
            label="D3 parameters",
            create=_unexpected_create,
        )

    assert target.read_bytes() == b"unexpected existing data"


def test_ensure_pinned_asset_does_not_publish_bad_generation(tmp_path: Path) -> None:
    target = tmp_path / "clean-runtime" / "dftd3" / "dftd3_parameters.pt"

    def create(staged_path: Path) -> None:
        staged_path.write_bytes(b"unexpected generated data")

    with pytest.raises(RuntimeError, match="D3 parameters.*expected.*found"):
        prewarm_assets.ensure_pinned_asset(
            target,
            expected_sha256=_sha256(b"expected generated data"),
            label="D3 parameters",
            create=create,
        )

    assert not target.exists()
    assert _staging_entries(target) == []


def test_generator_exception_cleans_partial_staging(tmp_path: Path) -> None:
    target = tmp_path / "clean-runtime" / "dftd3" / "dftd3_parameters.pt"
    staged_paths: list[Path] = []

    def create(staged_path: Path) -> NoReturn:
        staged_paths.append(staged_path)
        staged_path.write_bytes(b"partial generated data")
        raise RuntimeError("generator stopped")

    with pytest.raises(RuntimeError, match="generator stopped"):
        prewarm_assets.ensure_pinned_asset(
            target,
            expected_sha256=_sha256(b"complete generated data"),
            label="D3 parameters",
            create=create,
        )

    assert not target.exists()
    assert len(staged_paths) == 1
    assert not staged_paths[0].parent.exists()
    assert _staging_entries(target) == []


def test_matching_destination_appearing_during_generation_is_accepted(
    tmp_path: Path,
) -> None:
    payload = b"matching concurrent publication"
    target = tmp_path / "runtime" / "dftd3" / "dftd3_parameters.pt"
    winner_inodes: list[int] = []

    def create(staged_path: Path) -> None:
        staged_path.write_bytes(payload)
        target.write_bytes(payload)
        winner_inodes.append(target.stat().st_ino)

    result = prewarm_assets.ensure_pinned_asset(
        target,
        expected_sha256=_sha256(payload),
        label="D3 parameters",
        create=create,
    )

    assert result == target.resolve()
    assert target.read_bytes() == payload
    assert target.stat().st_ino == winner_inodes[0]
    assert _staging_entries(target) == []


def test_mismatched_destination_appearing_during_generation_is_preserved(
    tmp_path: Path,
) -> None:
    expected = b"verified staged publication"
    mismatch = b"concurrent mismatched winner"
    target = tmp_path / "runtime" / "dftd3" / "dftd3_parameters.pt"

    def create(staged_path: Path) -> None:
        staged_path.write_bytes(expected)
        target.write_bytes(mismatch)

    with pytest.raises(RuntimeError, match="D3 parameters.*expected.*found"):
        prewarm_assets.ensure_pinned_asset(
            target,
            expected_sha256=_sha256(expected),
            label="D3 parameters",
            create=create,
        )

    concurrent_mismatch_preserved = target.read_bytes() == mismatch
    assert concurrent_mismatch_preserved is True
    assert _staging_entries(target) == []


def test_two_concurrent_publishers_accept_one_complete_asset(tmp_path: Path) -> None:
    payload = b"same complete bytes from both publishers"
    target = tmp_path / "runtime" / "dftd3" / "dftd3_parameters.pt"
    creators_ready = threading.Barrier(2)

    def publish() -> Path:
        def create(staged_path: Path) -> None:
            staged_path.write_bytes(payload)
            creators_ready.wait(timeout=5)

        return prewarm_assets.ensure_pinned_asset(
            target,
            expected_sha256=_sha256(payload),
            label="D3 parameters",
            create=create,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: publish(), range(2)))

    assert results == [target.resolve(), target.resolve()]
    assert target.read_bytes() == payload
    assert _staging_entries(target) == []


def test_destination_is_invisible_until_verified_asset_is_complete(
    tmp_path: Path,
) -> None:
    partial = b"partial-"
    complete = partial + b"complete"
    target = tmp_path / "runtime" / "dftd3" / "dftd3_parameters.pt"
    partial_staged = threading.Event()
    finish_generation = threading.Event()

    def publish() -> Path:
        def create(staged_path: Path) -> None:
            staged_path.write_bytes(partial)
            partial_staged.set()
            assert finish_generation.wait(timeout=5)
            staged_path.write_bytes(complete)

        return prewarm_assets.ensure_pinned_asset(
            target,
            expected_sha256=_sha256(complete),
            label="D3 parameters",
            create=create,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(publish)
        assert partial_staged.wait(timeout=5)
        assert not target.exists()
        finish_generation.set()
        assert future.result(timeout=5) == target.resolve()

    assert target.read_bytes() == complete
    assert _staging_entries(target) == []


def test_atomic_no_clobber_unavailable_fails_clearly(tmp_path: Path) -> None:
    payload = b"verified but unpublished"
    target = tmp_path / "runtime" / "dftd3" / "dftd3_parameters.pt"

    def unavailable_link(_source: Path, _destination: Path) -> NoReturn:
        raise OSError(errno.EOPNOTSUPP, "hard links unavailable")

    with pytest.raises(
        RuntimeError, match="atomic no-clobber publication.*hard link.*unavailable"
    ):
        prewarm_assets.ensure_pinned_asset(
            target,
            expected_sha256=_sha256(payload),
            label="D3 parameters",
            create=lambda staged_path: staged_path.write_bytes(payload),
            publish_link=unavailable_link,
        )

    assert not target.exists()
    assert _staging_entries(target) == []


def test_runtime_check_rejects_missing_d3_parameter_file(tmp_path: Path) -> None:
    missing = tmp_path / "dftd3_parameters.pt"

    with pytest.raises(RuntimeError, match="D3 parameter file is missing"):
        check_runtime.require_file_sha256(
            "D3 parameter file",
            missing,
            prewarm_assets.PINS["dispersion"]["generated_parameter_sha256"],
        )


def test_runtime_check_rejects_mismatched_d3_parameter_file(tmp_path: Path) -> None:
    parameter_file = tmp_path / "dftd3_parameters.pt"
    parameter_file.write_bytes(b"not the pinned D3 parameters")

    with pytest.raises(
        RuntimeError, match="D3 parameter file SHA-256.*expected.*found"
    ):
        check_runtime.require_file_sha256(
            "D3 parameter file",
            parameter_file,
            prewarm_assets.PINS["dispersion"]["generated_parameter_sha256"],
        )


def test_main_preserves_aimnet_prepare_and_verify_behavior(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    alias = "test-aimnet-alias"
    checkpoint = tmp_path / "aimnet" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"verified AIMNet checkpoint")
    d3_path = tmp_path / "dftd3" / "dftd3_parameters.pt"
    d3_path.parent.mkdir()
    d3_path.write_bytes(b"verified D3 parameters")
    resolved_aliases: list[str] = []

    pins = deepcopy(prewarm_assets.PINS)
    pins["model"]["checkpoint_alias"] = alias
    pins["model"]["checkpoint_sha256"] = _sha256(checkpoint.read_bytes())
    pins["dispersion"]["generated_parameter_sha256"] = _sha256(d3_path.read_bytes())

    def resolve_model(requested_alias: str) -> Path:
        resolved_aliases.append(requested_alias)
        return checkpoint

    prewarm_assets.main(
        pins=pins,
        environment={"ALCHEMI_D3_PARAM_FILE": str(d3_path)},
        model_resolver=resolve_model,
    )

    assert resolved_aliases == [alias]
    assert f"AIMNet checkpoint verified: {checkpoint}" in capsys.readouterr().out
    assert checkpoint.read_bytes() == b"verified AIMNet checkpoint"


def test_main_rejects_aimnet_mismatch_before_d3_preparation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mismatched AIMNet checkpoint")
    d3_create_called = False

    pins = deepcopy(prewarm_assets.PINS)
    pins["model"]["checkpoint_sha256"] = _sha256(b"expected AIMNet checkpoint")

    def create_d3(_path: Path) -> None:
        nonlocal d3_create_called
        d3_create_called = True

    with pytest.raises(RuntimeError, match="AIMNet checkpoint.*expected.*found"):
        prewarm_assets.main(
            pins=pins,
            environment={},
            model_resolver=lambda _alias: checkpoint,
            d3_generator=create_d3,
        )

    assert d3_create_called is False
