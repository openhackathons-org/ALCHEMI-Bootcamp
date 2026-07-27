"""Strict loader for the small experimental water-fundamentals bundle.

The bundle contains six gas-phase fundamental wavenumbers transcribed from
Table 1 of Dinu et al., J. Phys. Chem. A 2019. It contains only the numeric
values, attribution, and source metadata, not a copied table, spectrum, or
intensity data. Loading verifies the local checksums and enforces the versioned
schema and exact six-row reference definition. It does not turn these markers
into a matched accuracy benchmark for MD or double-harmonic calculations.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import sha256_file


BUNDLE_SCHEMA = "alchemi.experimental-water-fundamentals.v2"
CHECKSUM_INDEX_NAME = "SHA256SUMS"
MANIFEST_NAME = "manifest.json"
DATA_NAME = "water_gas_phase_fundamentals.csv"

DEFAULT_BUNDLE_DIR = (
    Path(__file__).resolve().parents[1]
    / "reference"
    / "experimental_water_fundamentals"
)

EXPECTED_COLUMNS = (
    "isotopologue",
    "exact_isotopologue",
    "mode_index",
    "mode",
    "symmetry_species",
    "wavenumber_cm1",
    "units",
    "phase",
    "value_type",
    "source_id",
)

EXPECTED_ROWS = (
    (
        "H2O",
        "H2-16O",
        "1",
        "symmetric_stretch",
        "A1",
        "3657.1",
        "cm^-1",
        "gas",
        "observed_fundamental",
        "dinu_2019_table1_h2_16o",
    ),
    (
        "H2O",
        "H2-16O",
        "2",
        "bend",
        "A1",
        "1594.8",
        "cm^-1",
        "gas",
        "observed_fundamental",
        "dinu_2019_table1_h2_16o",
    ),
    (
        "H2O",
        "H2-16O",
        "3",
        "antisymmetric_stretch",
        "B2",
        "3755.9",
        "cm^-1",
        "gas",
        "observed_fundamental",
        "dinu_2019_table1_h2_16o",
    ),
    (
        "D2O",
        "D2-16O",
        "1",
        "symmetric_stretch",
        "A1",
        "2671.7",
        "cm^-1",
        "gas",
        "observed_fundamental",
        "dinu_2019_table1_d2_16o",
    ),
    (
        "D2O",
        "D2-16O",
        "2",
        "bend",
        "A1",
        "1178.4",
        "cm^-1",
        "gas",
        "observed_fundamental",
        "dinu_2019_table1_d2_16o",
    ),
    (
        "D2O",
        "D2-16O",
        "3",
        "antisymmetric_stretch",
        "B2",
        "2787.7",
        "cm^-1",
        "gas",
        "observed_fundamental",
        "dinu_2019_table1_d2_16o",
    ),
)

EXPECTED_CITATION = {
    "author_manuscript_url": (
        "https://aux.uibk.ac.at/c724117/publications/dinu19-jcp.pdf"
    ),
    "authors": [
        "Dennis F. Dinu",
        "Maren Podewitz",
        "Hinrich Grothe",
        "Klaus R. Liedl",
        "Thomas Loerting",
    ],
    "doi": "10.1021/acs.jpca.9b07221",
    "id": "dinu_2019",
    "license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "source_table": "Table 1",
    "text": (
        "Dinu, D. F.; Podewitz, M.; Grothe, H.; Liedl, K. R.; Loerting, T. "
        "Toward Elimination of Discrepancies between Theory and Experiment: "
        "Anharmonic Rotational-Vibrational Spectrum of Water in Solid Noble "
        "Gas Matrices. J. Phys. Chem. A 2019, 123, 8234-8242."
    ),
    "url": "https://doi.org/10.1021/acs.jpca.9b07221",
}

EXPECTED_SOURCES = {
    "dinu_2019_table1_h2_16o": {
        "article_citation_id": "dinu_2019",
        "display_label": "H2O",
        "exact_isotopologue": "H2-16O",
        "source_table": "Table 1",
        "underlying_citation_ids": [
            "toth_1999_h2_16o",
            "toth_1998_water_vapor",
        ],
    },
    "dinu_2019_table1_d2_16o": {
        "article_citation_id": "dinu_2019",
        "display_label": "D2O",
        "exact_isotopologue": "D2-16O",
        "source_table": "Table 1",
        "underlying_citation_ids": ["toth_1999_d2o"],
    },
}

EXPECTED_UNDERLYING_CITATIONS = {
    "toth_1998_water_vapor": {
        "doi": "10.1006/jmsp.1998.7611",
        "pages": "379-396",
        "pmid_url": "https://pubmed.ncbi.nlm.nih.gov/9668030/",
        "text": (
            "Toth, R. A. Water Vapor Measurements between 590 and 2582 cm^-1: "
            "Line Positions and Strengths. J. Mol. Spectrosc. 1998, 190(2), "
            "379-396."
        ),
        "title": (
            "Water Vapor Measurements between 590 and 2582 cm^-1: "
            "Line Positions and Strengths"
        ),
        "url": "https://doi.org/10.1006/jmsp.1998.7611",
        "volume_issue": "190(2)",
        "year": 1998,
    },
    "toth_1999_h2_16o": {
        "doi": "10.1006/jmsp.1998.7771",
        "pages": "28-42",
        "pmid_url": "https://pubmed.ncbi.nlm.nih.gov/9986772/",
        "text": (
            "Toth, R. A. Analysis of Line Positions and Strengths of H2-16O "
            "Ground and Hot Bands Connecting to Interacting Upper States: "
            "(020), (100), and (001). J. Mol. Spectrosc. 1999, 194(1), 28-42."
        ),
        "title": (
            "Analysis of Line Positions and Strengths of H2-16O Ground and Hot "
            "Bands Connecting to Interacting Upper States: (020), (100), and "
            "(001)"
        ),
        "url": "https://doi.org/10.1006/jmsp.1998.7771",
        "volume_issue": "194(1)",
        "year": 1999,
    },
    "toth_1999_d2o": {
        "doi": "10.1006/jmsp.1999.7815",
        "pages": "98-122",
        "pmid_url": "https://pubmed.ncbi.nlm.nih.gov/10191155/",
        "text": (
            "Toth, R. A. HDO and D2O Low Pressure, Long Path Spectra in the "
            "600-3100 cm^-1 Region: II. D2O Line Positions and Strengths. "
            "J. Mol. Spectrosc. 1999, 195(1), 98-122."
        ),
        "title": (
            "HDO and D2O Low Pressure, Long Path Spectra in the 600-3100 "
            "cm^-1 Region: II. D2O Line Positions and Strengths"
        ),
        "url": "https://doi.org/10.1006/jmsp.1999.7815",
        "volume_issue": "195(1)",
        "year": 1999,
    },
}

COMPARISON_BOUNDARY = (
    "These observed gas-phase fundamentals are reference markers, not a matched "
    "accuracy benchmark for a finite-temperature classical MD spectrum or a "
    "0 K double-harmonic calculation."
)
MODE_LABEL_CONVENTION = (
    "C2v labels follow Dinu et al. Table 1: the symmetric stretch and bend are "
    "A1 and the antisymmetric stretch is B2. A B1 label for the antisymmetric "
    "stretch uses the alternative convention that swaps the two in-plane axes; "
    "it does not describe a different vibration."
)
NO_REDISTRIBUTED_ARTICLE_CONTENT = (
    "No table image, article text, spectrum, or intensity data is redistributed."
)
TRANSCRIPTION_SCOPE = (
    "Only the six numeric gas-phase wavenumbers are transcribed from Dinu et al. "
    "Table 1."
)


class ExperimentalReferenceError(ValueError):
    """Raised when the experimental reference bundle is invalid."""


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentalReferenceError(f"manifest field {field!r} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], field: str
) -> None:
    observed = set(value)
    if observed == expected:
        return
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise ExperimentalReferenceError(
        f"manifest field {field!r} has the wrong keys ({'; '.join(details)})"
    )


def _load_checksum_index(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ExperimentalReferenceError(f"missing checksum index: {path}")
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ExperimentalReferenceError(
                f"malformed SHA256SUMS line {line_number}"
            )
        digest, relative_path = fields
        relative_path = relative_path.removeprefix("*")
        if len(digest) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in digest
        ):
            raise ExperimentalReferenceError(
                f"invalid SHA-256 on SHA256SUMS line {line_number}"
            )
        if relative_path in checksums:
            raise ExperimentalReferenceError(
                f"duplicate SHA256SUMS entry: {relative_path}"
            )
        checksums[relative_path] = digest.lower()

    expected = {MANIFEST_NAME, DATA_NAME}
    if set(checksums) != expected:
        raise ExperimentalReferenceError(
            "SHA256SUMS must cover exactly manifest.json and "
            "water_gas_phase_fundamentals.csv"
        )
    return checksums


def _verify_indexed_file(
    path: Path, *, relative_path: str, checksums: Mapping[str, str]
) -> str:
    if not path.is_file():
        raise ExperimentalReferenceError(f"missing bundle file: {path}")
    actual = sha256_file(path)
    if actual != checksums[relative_path]:
        raise ExperimentalReferenceError(f"SHA-256 mismatch for {relative_path}")
    return actual


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    data_path: Path,
    data_sha256: str,
) -> None:
    _require_exact_keys(
        manifest,
        {
            "artifact_id",
            "citation",
            "data",
            "integrity",
            "redistribution_boundary",
            "schema",
            "scope",
            "sources",
            "status",
            "underlying_citations",
            "units",
            "value_definition",
        },
        "root",
    )
    if manifest["schema"] != BUNDLE_SCHEMA:
        raise ExperimentalReferenceError(
            f"unexpected experimental reference schema: {manifest['schema']!r}"
        )
    if manifest["status"] != "complete":
        raise ExperimentalReferenceError("experimental reference status is not complete")
    expected_artifact_id = f"experimental-water-fundamentals-{data_sha256[:16]}"
    if manifest["artifact_id"] != expected_artifact_id:
        raise ExperimentalReferenceError("artifact_id does not match the data hash")

    data = _require_mapping(manifest["data"], "data")
    _require_exact_keys(
        data, {"bytes", "columns", "file", "row_count", "sha256"}, "data"
    )
    expected_data = {
        "bytes": data_path.stat().st_size,
        "columns": list(EXPECTED_COLUMNS),
        "file": DATA_NAME,
        "row_count": len(EXPECTED_ROWS),
        "sha256": data_sha256,
    }
    if dict(data) != expected_data:
        raise ExperimentalReferenceError("manifest data inventory is inconsistent")

    if manifest["citation"] != EXPECTED_CITATION:
        raise ExperimentalReferenceError("manifest citation is not the required source")
    sources = _require_mapping(manifest["sources"], "sources")
    if dict(sources) != EXPECTED_SOURCES:
        raise ExperimentalReferenceError(
            "manifest Table 1 sources or isotopologues are inconsistent"
        )
    underlying_citations = _require_mapping(
        manifest["underlying_citations"], "underlying_citations"
    )
    if dict(underlying_citations) != EXPECTED_UNDERLYING_CITATIONS:
        raise ExperimentalReferenceError(
            "manifest underlying Toth citations are inconsistent"
        )
    if manifest["units"] != {"wavenumber_cm1": "cm^-1"}:
        raise ExperimentalReferenceError("manifest wavenumber units must be cm^-1")
    if manifest["value_definition"] != {
        "exact_isotopologues": ["H2-16O", "D2-16O"],
        "mode_label_convention": MODE_LABEL_CONVENTION,
        "phase": "gas",
        "source_table": "Dinu et al. Table 1, gas-phase reference column",
        "transcription": TRANSCRIPTION_SCOPE,
        "type": "observed_fundamental",
    }:
        raise ExperimentalReferenceError(
            "manifest value definition must identify observed gas-phase fundamentals"
        )
    if manifest["integrity"] != {
        "checksum_index": CHECKSUM_INDEX_NAME,
        "checksum_index_covers_manifest": True,
    }:
        raise ExperimentalReferenceError("manifest integrity declaration is inconsistent")

    scope = _require_mapping(manifest["scope"], "scope")
    if scope.get("comparison_boundary") != COMPARISON_BOUNDARY:
        raise ExperimentalReferenceError(
            "manifest must preserve the non-benchmark comparison scope"
        )
    if scope.get("intensities_included") is not False:
        raise ExperimentalReferenceError("manifest must state that intensities are absent")
    if scope.get("spectrum_included") is not False:
        raise ExperimentalReferenceError("manifest must state that no spectrum is included")
    if scope.get("description") != (
        "Observed gas-phase fundamental wavenumbers for isolated H2-16O and "
        "D2-16O monomers, with H2O and D2O retained as display labels."
    ):
        raise ExperimentalReferenceError("manifest scope description is inconsistent")
    _require_exact_keys(
        scope,
        {
            "comparison_boundary",
            "description",
            "intensities_included",
            "spectrum_included",
        },
        "scope",
    )

    redistribution = _require_mapping(
        manifest["redistribution_boundary"], "redistribution_boundary"
    )
    _require_exact_keys(
        redistribution,
        {
            "included",
            "license",
            "license_url",
            "not_redistributed",
            "transcription",
        },
        "redistribution_boundary",
    )
    if redistribution.get("license") != "CC BY 4.0":
        raise ExperimentalReferenceError(
            "manifest must record the Dinu article's CC BY 4.0 license"
        )
    if redistribution.get("license_url") != (
        "https://creativecommons.org/licenses/by/4.0/"
    ):
        raise ExperimentalReferenceError(
            "manifest must link the CC BY 4.0 license"
        )
    if (
        redistribution.get("not_redistributed")
        != NO_REDISTRIBUTED_ARTICLE_CONTENT
    ):
        raise ExperimentalReferenceError(
            "manifest must state that article content and spectra are not redistributed"
        )
    if redistribution.get("transcription") != TRANSCRIPTION_SCOPE:
        raise ExperimentalReferenceError(
            "manifest must limit the transcription to the six numeric values"
        )


def _load_exact_table(data_path: Path) -> pd.DataFrame:
    try:
        raw = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ExperimentalReferenceError("experimental reference CSV is unreadable") from exc

    observed_columns = tuple(str(column) for column in raw.columns)
    if observed_columns != EXPECTED_COLUMNS:
        raise ExperimentalReferenceError(
            f"experimental reference columns are {observed_columns!r}; "
            f"expected {EXPECTED_COLUMNS!r}"
        )
    if len(raw) != len(EXPECTED_ROWS):
        raise ExperimentalReferenceError(
            f"experimental reference has {len(raw)} rows; expected {len(EXPECTED_ROWS)}"
        )

    observed_rows = tuple(
        tuple(str(value) for value in row)
        for row in raw.itertuples(index=False, name=None)
    )
    if observed_rows != EXPECTED_ROWS:
        mismatch = next(
            index
            for index, (observed, expected) in enumerate(
                zip(observed_rows, EXPECTED_ROWS, strict=True)
            )
            if observed != expected
        )
        differing_columns = [
            column
            for column, observed, expected in zip(
                EXPECTED_COLUMNS,
                observed_rows[mismatch],
                EXPECTED_ROWS[mismatch],
                strict=True,
            )
            if observed != expected
        ]
        raise ExperimentalReferenceError(
            f"experimental reference row {mismatch} has unexpected "
            + ", ".join(differing_columns)
        )

    table = raw.copy()
    table["mode_index"] = pd.to_numeric(table["mode_index"], errors="raise").astype(
        int
    )
    table["wavenumber_cm1"] = pd.to_numeric(
        table["wavenumber_cm1"], errors="raise"
    ).astype(float)
    return table


def load_experimental_water_fundamentals(
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
) -> pd.DataFrame:
    """Load the checksum-verified H2-16O/D2-16O gas-phase reference table.

    Checksums establish local bundle integrity, while the manifest and CSV
    checks enforce the exact Dinu Table 1 source, underlying Toth citations,
    isotopologues, units, phase, mode assignments, and observed values used by
    this tutorial. They do not independently reproduce the underlying
    experiments or make the markers a matched accuracy metric.
    """

    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise ExperimentalReferenceError(
            f"experimental reference bundle directory does not exist: {bundle_dir}"
        )

    checksums = _load_checksum_index(bundle_dir / CHECKSUM_INDEX_NAME)
    manifest_path = bundle_dir / MANIFEST_NAME
    data_path = bundle_dir / DATA_NAME
    _verify_indexed_file(
        manifest_path,
        relative_path=MANIFEST_NAME,
        checksums=checksums,
    )
    data_sha256 = _verify_indexed_file(
        data_path,
        relative_path=DATA_NAME,
        checksums=checksums,
    )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ExperimentalReferenceError(
            "experimental reference manifest is not valid JSON"
        ) from exc
    _validate_manifest(
        _require_mapping(manifest, "root"),
        data_path=data_path,
        data_sha256=data_sha256,
    )
    return _load_exact_table(data_path)
