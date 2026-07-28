"""Resolve, verify, and load the pinned SevenNet-Omni checkpoint."""

from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
from collections.abc import Callable
import sys
from typing import Any, TypeVar

import torch

from .sevennet_config import (
    SEVENNET_CHECKPOINT_BYTES,
    SEVENNET_CHECKPOINT_SHA256,
    SEVENNET_MODALITY,
    SEVENNET_MODEL_NAME,
)

_T = TypeVar("_T")
_SEVENNET_BACKEND_CONVERSION_LINE = "Converting model backend..."


def _run_without_backend_conversion_line(call: Callable[[], _T]) -> _T:
    """Run one SevenNet conversion without its expected status line.

    SevenNet prints this line while converting a checkpoint to the requested
    backend. Any other stdout from the same call is written back unchanged.
    """

    captured = StringIO()
    try:
        with redirect_stdout(captured):
            return call()
    finally:
        unexpected = "".join(
            line
            for line in captured.getvalue().splitlines(keepends=True)
            if line.rstrip("\r\n") != _SEVENNET_BACKEND_CONVERSION_LINE
        )
        if unexpected:
            sys.stdout.write(unexpected)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of *path*."""

    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_sevennet_checkpoint(
    *,
    path_resolver: Callable[[str], str | Path] | None = None,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    digest_reader: Callable[[str | Path], str] = sha256_file,
) -> tuple[Path, str]:
    """Resolve SevenNet's official asset and verify its size and digest."""

    if path_resolver is None:
        from sevenn.util import pretrained_name_to_path

        path_resolver = pretrained_name_to_path
    if expected_bytes is None:
        expected_bytes = SEVENNET_CHECKPOINT_BYTES
    if expected_sha256 is None:
        expected_sha256 = SEVENNET_CHECKPOINT_SHA256

    path = Path(path_resolver(SEVENNET_MODEL_NAME)).resolve()
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(
            "SevenNet-Omni checkpoint size changed: "
            f"expected {expected_bytes:,} bytes, "
            f"found {path.stat().st_size:,}"
        )
    digest = digest_reader(path)
    if digest != expected_sha256:
        raise RuntimeError(
            "SevenNet-Omni checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, found {digest}"
        )
    return path, digest


def load_raw_sevennet_omni(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    checkpoint_loader: Callable[[Path], Any] | None = None,
    required_modality: str = SEVENNET_MODALITY,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the raw float32 e3nn model used by the teaching adapter."""

    if checkpoint_loader is None:
        from sevenn.util import load_checkpoint

        checkpoint_loader = load_checkpoint

    checkpoint = checkpoint_loader(Path(checkpoint_path))
    raw_model = _run_without_backend_conversion_line(
        lambda: checkpoint.build_model(
            enable_cueq=False,
            enable_flash=False,
            enable_oeq=False,
        )
    )
    if required_modality not in getattr(raw_model, "modal_map", {}):
        raise RuntimeError(
            f"{SEVENNET_MODEL_NAME} does not expose the "
            f"{required_modality!r} task"
        )
    raw_model = raw_model.to(device=device, dtype=torch.float32).eval()
    for parameter in raw_model.parameters():
        parameter.requires_grad_(False)
    return raw_model, dict(checkpoint.config)


__all__ = [
    "load_raw_sevennet_omni",
    "resolve_sevennet_checkpoint",
    "sha256_file",
]
