"""Keep SevenNet lesson settings aligned with image prewarming."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PART_DIR))

from build import prewarm_sevennet  # noqa: E402
from aux.models.sevennet_config import (  # noqa: E402
    BOHR_TO_ANGSTROM,
    D3_REFERENCE_CUTOFF_A,
    D3_REFERENCE_CUTOFF_BOHR,
    D3_REFERENCE_SMOOTHING_FRACTION,
    PBE_D3_BJ_A1,
    PBE_D3_BJ_A2_BOHR,
    PBE_D3_BJ_S6,
    PBE_D3_BJ_S8,
    SEVENNET_CHECKPOINT_BYTES,
    SEVENNET_CHECKPOINT_SHA256,
    SEVENNET_CHECKPOINT_URL,
    SEVENNET_MODALITY,
    SEVENNET_MODEL_NAME,
    SEVENNET_PACKAGE_VERSION,
    SEVENNET_REFERENCE_METHOD,
    SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM,
    SEVENNET_REPEAT_FORCE_TOL_EV_A,
)


def test_adapter_and_prewarm_use_the_same_model_identity() -> None:
    assert SEVENNET_CHECKPOINT_URL == prewarm_sevennet.SEVENNET_CHECKPOINT_URL
    assert SEVENNET_CHECKPOINT_BYTES == prewarm_sevennet.SEVENNET_CHECKPOINT_BYTES
    assert (
        SEVENNET_CHECKPOINT_SHA256
        == prewarm_sevennet.SEVENNET_CHECKPOINT_SHA256
    )
    assert SEVENNET_MODEL_NAME == prewarm_sevennet.SEVENNET_MODEL_NAME
    assert SEVENNET_PACKAGE_VERSION == prewarm_sevennet.SEVENNET_PACKAGE_VERSION
    assert SEVENNET_MODALITY == prewarm_sevennet.SEVENNET_MODALITY


def test_model_and_d3_settings_are_explicit_and_internally_consistent() -> None:
    assert SEVENNET_MODEL_NAME == "7net-omni"
    assert SEVENNET_MODALITY == "mpa"
    assert "PBE(+U)" in SEVENNET_REFERENCE_METHOD
    assert "no D3" in SEVENNET_REFERENCE_METHOD
    assert len(SEVENNET_CHECKPOINT_SHA256) == 64
    int(SEVENNET_CHECKPOINT_SHA256, 16)
    assert SEVENNET_CHECKPOINT_BYTES > 0
    assert SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM > 0.0
    assert SEVENNET_REPEAT_FORCE_TOL_EV_A > 0.0
    assert PBE_D3_BJ_A1 > 0.0
    assert PBE_D3_BJ_A2_BOHR > 0.0
    assert PBE_D3_BJ_S6 > 0.0
    assert PBE_D3_BJ_S8 > 0.0
    assert D3_REFERENCE_CUTOFF_A == pytest.approx(
        D3_REFERENCE_CUTOFF_BOHR * BOHR_TO_ANGSTROM
    )
    assert D3_REFERENCE_SMOOTHING_FRACTION == 0.0


def _sevenn_constants(checkpoint_url: str) -> SimpleNamespace:
    return SimpleNamespace(
        SEVENNET_omni="omni",
        CHECKPOINT_DOWNLOAD_LINKS={"omni": checkpoint_url},
    )


def test_prewarm_resolver_accepts_only_the_pinned_upstream_asset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint_sevennet_omni.pth"
    path.write_bytes(b"checkpoint")

    resolved, digest = prewarm_sevennet.resolve_checkpoint(
        version_reader=lambda _package: SEVENNET_PACKAGE_VERSION,
        sevenn_constants=_sevenn_constants(SEVENNET_CHECKPOINT_URL),
        path_resolver=lambda _name: path,
        expected_bytes=path.stat().st_size,
        digest_reader=lambda _path: SEVENNET_CHECKPOINT_SHA256,
    )

    assert resolved == path.resolve()
    assert digest == SEVENNET_CHECKPOINT_SHA256


@pytest.mark.parametrize(
    ("version", "url", "message"),
    [
        ("0.12.0", SEVENNET_CHECKPOINT_URL, "expected sevenn"),
        (SEVENNET_PACKAGE_VERSION, "https://example.invalid/model.pth", "URL changed"),
    ],
)
def test_prewarm_resolver_rejects_package_or_asset_drift(
    tmp_path: Path,
    version: str,
    url: str,
    message: str,
) -> None:
    path = tmp_path / "checkpoint_sevennet_omni.pth"
    path.write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError, match=message):
        prewarm_sevennet.resolve_checkpoint(
            version_reader=lambda _package: version,
            sevenn_constants=_sevenn_constants(url),
            path_resolver=lambda _name: path,
        )


def test_prewarm_requires_the_custom_toolkit_adapter(tmp_path: Path) -> None:
    """An official-calculator pass may not hide a broken tutorial adapter."""

    checkpoint = Path("checkpoint_sevennet_omni.pth")
    calls: list[tuple[str, Path, str, Path | None]] = []

    def official(path: Path, *, device: str):
        calls.append(("official", path, device, None))
        return ({"energy_eV": -1.0}, object())

    def toolkit(
        path: Path,
        *,
        device: str,
        source_root: Path | None = None,
    ):
        calls.append(("toolkit", path, device, source_root))
        raise RuntimeError("custom adapter failed")

    with pytest.raises(RuntimeError, match="custom adapter failed"):
        prewarm_sevennet.smoke_test(
            checkpoint,
            device="cuda",
            source_root=tmp_path,
            official_smoke=official,
            toolkit_smoke=toolkit,
        )

    assert calls == [
        ("official", checkpoint, "cuda", None),
        ("toolkit", checkpoint, "cuda", tmp_path),
    ]
