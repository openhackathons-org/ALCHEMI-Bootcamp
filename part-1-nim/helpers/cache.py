"""Cache layer for FAST_DEMO mode — serialise/deserialise Pydantic response objects."""

import json
import os
from pathlib import Path

from .models import BGRReply, BMDReply


def _cache_path(cache_dir: str, label: str, ext: str = ".json") -> Path:
    return Path(cache_dir) / f"{label}{ext}"


def cache_exists(cache_dir: str, label: str) -> bool:
    """Return True if a cached response file exists for *label*."""
    return _cache_path(cache_dir, label).is_file()


def save_cache(cache_dir: str, label: str, reply: BMDReply | BGRReply) -> Path:
    """Serialise a Pydantic reply to JSON and return the file path."""
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, label)
    path.write_text(reply.model_dump_json(indent=2))
    return path


def load_cache(
    cache_dir: str,
    label: str,
    model_cls: type[BMDReply] | type[BGRReply],
) -> BMDReply | BGRReply:
    """Deserialise a cached JSON file back into *model_cls*."""
    path = _cache_path(cache_dir, label)
    data = json.loads(path.read_text())
    return model_cls(**data)
