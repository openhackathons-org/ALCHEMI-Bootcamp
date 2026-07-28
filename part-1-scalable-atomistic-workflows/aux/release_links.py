"""Local notebook links that must remain valid in a reviewed release copy."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit


LOCAL_NOTEBOOK_REFERENCES = (
    "assets/images/banner_candidates/water-ir-v2-04-trajectory-to-spectrum.png",
    "COMPUTE_LAB_RUNBOOK.md#5-build-and-check-the-recorded-result-set",
    "../part-2-batched-adsorption-toolkit/README.md",
    "../THIRD_PARTY_NOTICES.md",
)


def _local_parts(reference: str) -> SplitResult:
    """Parse one declared local link and reject non-file references."""

    parts = urlsplit(reference)
    if parts.scheme or parts.netloc or not parts.path:
        raise ValueError(f"not a local file reference: {reference!r}")
    return parts


def local_reference_replacements(
    *,
    source_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Map source-relative links to paths relative to a reviewed notebook."""

    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    replacements: dict[str, str] = {}
    for reference in LOCAL_NOTEBOOK_REFERENCES:
        parts = _local_parts(reference)
        target = (source_dir / parts.path).resolve()
        if not target.exists():
            raise FileNotFoundError(
                f"local notebook reference does not exist: {reference!r}"
            )
        relative_path = Path(os.path.relpath(target, output_dir)).as_posix()
        replacements[reference] = urlunsplit(
            ("", "", relative_path, parts.query, parts.fragment)
        )
    return replacements


__all__ = ["LOCAL_NOTEBOOK_REFERENCES", "local_reference_replacements"]
