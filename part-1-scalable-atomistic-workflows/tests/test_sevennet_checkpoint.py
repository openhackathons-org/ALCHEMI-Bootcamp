"""Focused tests for pinned SevenNet checkpoint loading."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")

PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.models import sevennet_checkpoint  # noqa: E402
from aux.models.sevennet_checkpoint import (  # noqa: E402
    load_raw_sevennet_omni,
    resolve_sevennet_checkpoint,
    sha256_file,
)


def test_sha256_file_streams_the_complete_file(tmp_path: Path) -> None:
    payload = b"a" * (1024 * 1024 + 17) + b"last chunk"
    path = tmp_path / "checkpoint.pth"
    path.write_bytes(payload)

    assert sha256_file(path) == sha256(payload).hexdigest()


def test_resolve_checkpoint_verifies_size_and_digest(
    tmp_path: Path,
) -> None:
    payload = b"pinned SevenNet checkpoint"
    path = tmp_path / "checkpoint_sevennet_omni.pth"
    path.write_bytes(payload)
    calls = []

    def resolve(name: str) -> str:
        calls.append(name)
        return str(path)

    resolved, digest = resolve_sevennet_checkpoint(
        path_resolver=resolve,
        expected_bytes=len(payload),
        expected_sha256=sha256(payload).hexdigest(),
    )

    assert calls == [sevennet_checkpoint.SEVENNET_MODEL_NAME]
    assert resolved == path.resolve()
    assert digest == sha256(payload).hexdigest()


@pytest.mark.parametrize("failure", ["size", "digest"])
def test_resolve_checkpoint_rejects_changed_assets(
    tmp_path: Path,
    failure: str,
) -> None:
    payload = b"changed asset"
    path = tmp_path / "checkpoint.pth"
    path.write_bytes(payload)
    expected_digest = (
        "0" * 64 if failure == "digest" else sha256(payload).hexdigest()
    )

    with pytest.raises(RuntimeError, match="size changed|SHA-256 mismatch"):
        resolve_sevennet_checkpoint(
            path_resolver=lambda _name: path,
            expected_bytes=len(payload) + (1 if failure == "size" else 0),
            expected_sha256=expected_digest,
        )


class _FakeRawModel(torch.nn.Module):
    def __init__(self, *, modalities: tuple[str, ...] = ("mpa",)) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.modal_map = {name: index for index, name in enumerate(modalities)}


class _FakeCheckpoint:
    def __init__(self, raw_model: _FakeRawModel) -> None:
        self.raw_model = raw_model
        self.config = {"cutoff": 5.0, "task": "mpa"}
        self.build_calls = []

    def build_model(self, **kwargs):
        self.build_calls.append(kwargs)
        return self.raw_model


class _ChattyCheckpoint(_FakeCheckpoint):
    def build_model(self, **kwargs):
        print("Converting model backend...")
        print("unexpected checkpoint output")
        return super().build_model(**kwargs)


def test_load_raw_model_uses_plain_e3nn_and_freezes_parameters() -> None:
    raw_model = _FakeRawModel()
    checkpoint = _FakeCheckpoint(raw_model)
    loaded_paths = []

    def load(path: Path):
        loaded_paths.append(path)
        return checkpoint

    loaded, config = load_raw_sevennet_omni(
        "model.pth",
        device="cpu",
        checkpoint_loader=load,
    )

    assert loaded_paths == [Path("model.pth")]
    assert checkpoint.build_calls == [
        {"enable_cueq": False, "enable_flash": False, "enable_oeq": False}
    ]
    assert loaded is raw_model
    assert not loaded.training
    assert raw_model.weight.dtype is torch.float32
    assert raw_model.weight.device.type == "cpu"
    assert not raw_model.weight.requires_grad
    assert config == checkpoint.config
    assert config is not checkpoint.config


def test_load_raw_model_hides_only_expected_conversion_line(capsys) -> None:
    checkpoint = _ChattyCheckpoint(_FakeRawModel())

    load_raw_sevennet_omni(
        "model.pth",
        device="cpu",
        checkpoint_loader=lambda _path: checkpoint,
    )

    assert capsys.readouterr().out == "unexpected checkpoint output\n"


def test_load_raw_model_requires_the_configured_task() -> None:
    checkpoint = _FakeCheckpoint(_FakeRawModel(modalities=("oc20",)))

    with pytest.raises(RuntimeError, match="does not expose.*'mpa'"):
        load_raw_sevennet_omni(
            "model.pth",
            device="cpu",
            checkpoint_loader=lambda _path: checkpoint,
        )
