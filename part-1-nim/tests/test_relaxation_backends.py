"""Backend selection tests for BGR NIM and native Toolkit relaxation."""

from __future__ import annotations

import importlib

import pytest

from helpers.cache import save_cache
from helpers.models import BGRReply, OptimizationResult
from helpers.relaxation_backends import (
    BackendUnavailableError,
    BgrNimBackend,
    RelaxationBackendConfig,
    get_relaxation_backend,
    require_toolkit_api,
)


def test_bgr_backend_selection_is_explicit():
    backend = get_relaxation_backend(RelaxationBackendConfig(name="bgr_nim"))
    assert isinstance(backend, BgrNimBackend)
    assert backend.name == "bgr_nim"


def test_unknown_backend_rejected():
    config = RelaxationBackendConfig(name="bgr_nim")
    object.__setattr__(config, "name", "ase_fallback")

    with pytest.raises(ValueError, match="Unknown relaxation backend"):
        get_relaxation_backend(config)


def test_toolkit_import_failure_message_is_precise(monkeypatch):
    real_import_module = importlib.import_module

    def blocked_import(name: str, *args, **kwargs):
        if name == "torch":
            raise ImportError("blocked for test", name=name)
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", blocked_import)

    with pytest.raises(BackendUnavailableError) as excinfo:
        require_toolkit_api()

    message = str(excinfo.value)
    assert "ADSORPTION_BACKEND='toolkit' selected" in message
    assert "torch" in message
    assert "No BGR fallback was attempted" in message


def test_toolkit_cache_replay_skips_toolkit_import(monkeypatch, tmp_path):
    real_import_module = importlib.import_module

    def blocked_import(name: str, *args, **kwargs):
        if name == "torch":
            raise AssertionError("Toolkit cache replay should not import torch")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", blocked_import)

    cached = BGRReply(
        atoms=[
            OptimizationResult(
                coord=[0.0, 0.0, 0.0],
                numbers=[1],
                converged=True,
                optimizer_nsteps=0,
                energy=-1.0,
                forces=[0.0, 0.0, 0.0],
            )
        ],
        status="Success",
        info="cached toolkit response",
    )
    save_cache(str(tmp_path), "cached_case", cached)

    backend = get_relaxation_backend(
        RelaxationBackendConfig(
            name="toolkit",
            cache_dir=str(tmp_path),
            use_cached_responses=True,
        )
    )

    loaded = backend.relax([], label="cached_case")

    assert loaded.info == "cached toolkit response"
    assert loaded.atoms[0].energy == -1.0
