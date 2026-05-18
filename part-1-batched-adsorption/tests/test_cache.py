"""Test caching layer."""

import os
import tempfile
from pathlib import Path

import pytest

from helpers.cache import cache_exists, load_cache, save_cache
from helpers.cache_registry import (
    LATEST_COMPLETE_RUN_ID,
    SURFACE_SCREEN_REQUIRED_FILES,
    find_latest_complete_live_run,
    make_live_run_id,
    require_complete,
    resolve_run_roots,
    validate_surface_screen,
)
from helpers.models import OptimizationResult, RelaxationBatchResult


class TestCache:
    @pytest.fixture
    def tmp_cache_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def sample_reply(self):
        result = OptimizationResult(
            coord=[1.0, 2.0, 3.0],
            numbers=[1],
            converged=True,
            optimizer_nsteps=10,
            energy=-100.5,
            forces=[0.0, 0.0, 0.0],
        )
        return RelaxationBatchResult(atoms=[result], status="Success")

    def test_cache_roundtrip(self, tmp_cache_dir, sample_reply):
        save_cache(tmp_cache_dir, "test_run", sample_reply)
        assert cache_exists(tmp_cache_dir, "test_run")

        loaded = load_cache(tmp_cache_dir, "test_run", RelaxationBatchResult)
        assert loaded.status == "Success"
        assert len(loaded.atoms) == 1
        assert loaded.atoms[0].energy == pytest.approx(-100.5)

    def test_cache_not_exists(self, tmp_cache_dir):
        assert not cache_exists(tmp_cache_dir, "nonexistent")

    def test_save_creates_directory(self, tmp_cache_dir, sample_reply):
        nested = os.path.join(tmp_cache_dir, "sub", "dir")
        save_cache(nested, "test_run", sample_reply)
        assert cache_exists(nested, "test_run")

    def test_cache_json_format(self, tmp_cache_dir, sample_reply):
        import json

        path = save_cache(tmp_cache_dir, "test_run", sample_reply)
        data = json.loads(path.read_text())
        assert "atoms" in data
        assert "status" in data

    def test_save_cache_refuses_accidental_overwrite(self, tmp_cache_dir, sample_reply):
        save_cache(tmp_cache_dir, "test_run", sample_reply)
        with pytest.raises(FileExistsError):
            save_cache(tmp_cache_dir, "test_run", sample_reply)

    def test_save_cache_allows_intentional_overwrite(self, tmp_cache_dir, sample_reply):
        save_cache(tmp_cache_dir, "test_run", sample_reply)
        path = save_cache(tmp_cache_dir, "test_run", sample_reply, overwrite=True)
        assert path.is_file()


class TestRunRegistry:
    @staticmethod
    def _write_required_files(root: Path, required_files):
        for name in required_files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ok", encoding="utf-8")

    def test_resolve_official_saved_roots(self, tmp_path: Path):
        roots = resolve_run_roots(
            tutorial_root=tmp_path,
            run_scope="full",
            use_saved_tutorial_results=True,
            use_saved_accuracy_results=True,
            saved_tutorial_run_id=None,
            saved_accuracy_run_id=None,
            refresh_saved_results=False,
            live_run_id="20260518-120000",
        )
        assert roots.tutorial_output_dir == Path("outputs/precomputed/tutorial")
        assert roots.accuracy_output_dir == Path("outputs/precomputed/accuracy")
        assert roots.surface_screen_root == Path(
            "outputs/precomputed/tutorial/surface_screen_v1_mh1_oc20_usemppbe_full/surface_screen"
        )

    def test_resolve_explicit_live_run(self, tmp_path: Path):
        roots = resolve_run_roots(
            tutorial_root=tmp_path,
            run_scope="short",
            use_saved_tutorial_results=True,
            use_saved_accuracy_results=True,
            saved_tutorial_run_id="20260518-120000",
            saved_accuracy_run_id="20260518-130000",
            refresh_saved_results=False,
            live_run_id="20260518-140000",
        )
        assert roots.tutorial_output_dir == Path("outputs/live_runs/20260518-120000/tutorial")
        assert roots.accuracy_output_dir == Path("outputs/live_runs/20260518-130000/accuracy")
        assert "selected live run 20260518-120000" == roots.tutorial_source_label

    def test_latest_complete_live_run_requires_complete_artifacts(self, tmp_path: Path):
        incomplete = (
            tmp_path
            / "outputs"
            / "live_runs"
            / "20260518-120000"
            / "tutorial"
            / "surface_screen_v1_mh1_oc20_usemppbe_full"
            / "surface_screen"
        )
        incomplete.mkdir(parents=True)
        complete = (
            tmp_path
            / "outputs"
            / "live_runs"
            / "20260518-130000"
            / "tutorial"
            / "surface_screen_v1_mh1_oc20_usemppbe_full"
            / "surface_screen"
        )
        self._write_required_files(complete, SURFACE_SCREEN_REQUIRED_FILES)

        assert (
            find_latest_complete_live_run(
                tmp_path,
                run_scope="full",
                artifact_kind="tutorial",
            )
            == "20260518-130000"
        )

        roots = resolve_run_roots(
            tutorial_root=tmp_path,
            run_scope="full",
            use_saved_tutorial_results=True,
            use_saved_accuracy_results=False,
            saved_tutorial_run_id=LATEST_COMPLETE_RUN_ID,
            saved_accuracy_run_id=None,
            refresh_saved_results=False,
            live_run_id="20260518-140000",
        )
        assert roots.tutorial_output_dir == Path("outputs/live_runs/20260518-130000/tutorial")

    def test_validate_surface_screen_reports_missing_files(self, tmp_path: Path):
        root = tmp_path / "surface_screen"
        root.mkdir()
        validation = validate_surface_screen(root)
        assert not validation.complete
        assert set(validation.missing) == set(SURFACE_SCREEN_REQUIRED_FILES)
        with pytest.raises(FileNotFoundError, match="Saved surface screen cache is incomplete"):
            require_complete(validation, label="surface screen")

    def test_make_live_run_id_is_sortable(self):
        assert make_live_run_id().__len__() == len("20260518-120000")
