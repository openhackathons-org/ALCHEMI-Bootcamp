"""Cache layer for tutorial Pydantic response objects."""

import json
import os
from pathlib import Path

from .models import RelaxationBatchResult

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _cache_path(cache_dir: str, label: str, ext: str = ".json") -> Path:
    return Path(cache_dir) / f"{label}{ext}"


def _env_allows_cache_overwrite() -> bool:
    return (
        os.environ.get("ALCHEMI_ALLOW_CACHE_OVERWRITE", "").strip().lower()
        in _TRUE_VALUES
    )


def cache_exists(cache_dir: str, label: str) -> bool:
    """Return True if a cached response file exists for *label*."""
    return _cache_path(cache_dir, label).is_file()


def save_cache(
    cache_dir: str,
    label: str,
    reply: RelaxationBatchResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Serialise a Pydantic reply to JSON and return the file path."""
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, label)
    if path.exists() and not (overwrite or _env_allows_cache_overwrite()):
        raise FileExistsError(
            f"Cached response already exists: {path}. "
            "Use a live-run cache directory, pass overwrite=True, or set "
            "ALCHEMI_ALLOW_CACHE_OVERWRITE=1 for an intentional refresh."
        )
    path.write_text(reply.model_dump_json(indent=2))
    return path


def load_cache(
    cache_dir: str,
    label: str,
    model_cls: type[RelaxationBatchResult],
) -> RelaxationBatchResult:
    """Deserialise a cached JSON file back into *model_cls*."""
    path = _cache_path(cache_dir, label)
    data = json.loads(path.read_text())
    return model_cls(**data)
