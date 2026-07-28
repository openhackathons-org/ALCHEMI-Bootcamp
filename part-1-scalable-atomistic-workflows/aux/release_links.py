"""Local notebook files and links used by the portable learner-review copy."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit

PACKAGED_NOTEBOOK_ASSETS = (
    "assets/images/banner_candidates/water-ir-v2-04-trajectory-to-spectrum.png",
)

PACKAGED_NOTEBOOK_DOCUMENTS = (
    (
        "COMPUTE_LAB_RUNBOOK.md",
        "docs/part-1-scalable-atomistic-workflows/COMPUTE_LAB_RUNBOOK.md",
    ),
    (
        "../part-2-batched-adsorption-toolkit/README.md",
        "docs/part-2-batched-adsorption-toolkit/README.md",
    ),
    ("../THIRD_PARTY_NOTICES.md", "docs/THIRD_PARTY_NOTICES.md"),
    (
        "reference/README.md",
        "docs/part-1-scalable-atomistic-workflows/reference/README.md",
    ),
    (
        "data/nci_atlas/README.md",
        "docs/part-1-scalable-atomistic-workflows/data/nci_atlas/README.md",
    ),
)

PACKAGED_DOCUMENT_LINK_REPLACEMENTS = {
    "docs/part-2-batched-adsorption-toolkit/README.md": {
        "../part-1-scalable-atomistic-workflows/": (
            "../part-1-scalable-atomistic-workflows/COMPUTE_LAB_RUNBOOK.md"
        ),
    },
}

PACKAGED_NOTEBOOK_FILES = (
    *((reference, reference) for reference in PACKAGED_NOTEBOOK_ASSETS),
    *PACKAGED_NOTEBOOK_DOCUMENTS,
)

LOCAL_NOTEBOOK_LINKS = (
    "COMPUTE_LAB_RUNBOOK.md#5-build-and-check-the-recorded-result-set",
    "../part-2-batched-adsorption-toolkit/README.md",
    "../THIRD_PARTY_NOTICES.md",
)

LOCAL_NOTEBOOK_REFERENCES = (*PACKAGED_NOTEBOOK_ASSETS, *LOCAL_NOTEBOOK_LINKS)

LOCAL_NOTEBOOK_OUTPUT_REFERENCES = {
    PACKAGED_NOTEBOOK_ASSETS[0]: PACKAGED_NOTEBOOK_ASSETS[0],
    LOCAL_NOTEBOOK_LINKS[0]: (
        "docs/part-1-scalable-atomistic-workflows/COMPUTE_LAB_RUNBOOK.md"
        "#5-build-and-check-the-recorded-result-set"
    ),
    LOCAL_NOTEBOOK_LINKS[1]: ("docs/part-2-batched-adsorption-toolkit/README.md"),
    LOCAL_NOTEBOOK_LINKS[2]: "docs/THIRD_PARTY_NOTICES.md",
}


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
        if not target.is_file():
            raise FileNotFoundError(
                f"local notebook reference is not a file: {reference!r}"
            )
        packaged_reference = LOCAL_NOTEBOOK_OUTPUT_REFERENCES[reference]
        packaged_parts = _local_parts(packaged_reference)
        packaged_target = (output_dir / packaged_parts.path).resolve()
        try:
            packaged_target.relative_to(output_dir)
        except ValueError as error:
            raise RuntimeError(
                f"packaged notebook reference escapes the output directory: "
                f"{packaged_reference!r}"
            ) from error
        replacements[reference] = urlunsplit(packaged_parts)
    return replacements


__all__ = [
    "LOCAL_NOTEBOOK_LINKS",
    "LOCAL_NOTEBOOK_OUTPUT_REFERENCES",
    "LOCAL_NOTEBOOK_REFERENCES",
    "PACKAGED_DOCUMENT_LINK_REPLACEMENTS",
    "PACKAGED_NOTEBOOK_ASSETS",
    "PACKAGED_NOTEBOOK_DOCUMENTS",
    "PACKAGED_NOTEBOOK_FILES",
    "local_reference_replacements",
]
