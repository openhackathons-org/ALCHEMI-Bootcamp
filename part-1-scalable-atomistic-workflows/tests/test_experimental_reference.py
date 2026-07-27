"""Tests for the source-attributed experimental water reference bundle."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.artifacts import sha256_file  # noqa: E402
from aux.experimental_reference import (  # noqa: E402
    COMPARISON_BOUNDARY,
    DATA_NAME,
    DEFAULT_BUNDLE_DIR,
    EXPECTED_COLUMNS,
    EXPECTED_SOURCES,
    EXPECTED_UNDERLYING_CITATIONS,
    MODE_LABEL_CONVENTION,
    TRANSCRIPTION_SCOPE,
    ExperimentalReferenceError,
    load_experimental_water_fundamentals,
)


def _copy_bundle(tmp_path: Path) -> Path:
    destination = tmp_path / "experimental_water_fundamentals"
    shutil.copytree(DEFAULT_BUNDLE_DIR, destination)
    return destination


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _seal_bundle(bundle_dir: Path, manifest: dict[str, Any]) -> None:
    data_path = bundle_dir / DATA_NAME
    data_sha256 = sha256_file(data_path)
    manifest["artifact_id"] = f"experimental-water-fundamentals-{data_sha256[:16]}"
    manifest["data"]["bytes"] = data_path.stat().st_size
    manifest["data"]["sha256"] = data_sha256
    manifest_path = bundle_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    (bundle_dir / "SHA256SUMS").write_text(
        f"{sha256_file(manifest_path)}  manifest.json\n"
        f"{data_sha256}  {DATA_NAME}\n",
        encoding="utf-8",
    )


def _set_nested(root: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = root
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_loads_exact_six_observed_gas_phase_fundamentals() -> None:
    table = load_experimental_water_fundamentals()

    assert tuple(table.columns) == EXPECTED_COLUMNS
    assert table["isotopologue"].tolist() == ["H2O"] * 3 + ["D2O"] * 3
    assert table["exact_isotopologue"].tolist() == (
        ["H2-16O"] * 3 + ["D2-16O"] * 3
    )
    assert table["mode_index"].tolist() == [1, 2, 3, 1, 2, 3]
    assert table["mode"].tolist() == [
        "symmetric_stretch",
        "bend",
        "antisymmetric_stretch",
        "symmetric_stretch",
        "bend",
        "antisymmetric_stretch",
    ]
    assert table["symmetry_species"].tolist() == ["A1", "A1", "B2"] * 2
    assert table["wavenumber_cm1"].tolist() == [
        3657.1,
        1594.8,
        3755.9,
        2671.7,
        1178.4,
        2787.7,
    ]
    assert set(table["units"]) == {"cm^-1"}
    assert set(table["phase"]) == {"gas"}
    assert set(table["value_type"]) == {"observed_fundamental"}
    assert "intensity" not in " ".join(table.columns).casefold()


def test_manifest_records_exact_sources_and_nonbenchmark_boundary() -> None:
    manifest_text = (DEFAULT_BUNDLE_DIR / "manifest.json").read_text()
    manifest = json.loads(manifest_text)

    assert manifest["sources"] == EXPECTED_SOURCES
    assert manifest["underlying_citations"] == EXPECTED_UNDERLYING_CITATIONS
    assert manifest["citation"]["doi"] == "10.1021/acs.jpca.9b07221"
    assert manifest["citation"]["source_table"] == "Table 1"
    assert manifest["citation"]["license"] == "CC BY 4.0"
    assert manifest["citation"]["license_url"] == (
        "https://creativecommons.org/licenses/by/4.0/"
    )
    assert manifest["value_definition"]["exact_isotopologues"] == [
        "H2-16O",
        "D2-16O",
    ]
    assert (
        manifest["value_definition"]["mode_label_convention"]
        == MODE_LABEL_CONVENTION
    )
    assert manifest["value_definition"]["transcription"] == TRANSCRIPTION_SCOPE
    assert manifest["scope"] == {
        "comparison_boundary": COMPARISON_BOUNDARY,
        "description": (
            "Observed gas-phase fundamental wavenumbers for isolated H2-16O "
            "and D2-16O monomers, with H2O and D2O retained as display labels."
        ),
        "intensities_included": False,
        "spectrum_included": False,
    }
    assert manifest["redistribution_boundary"]["not_redistributed"] == (
        "No table image, article text, spectrum, or intensity data is "
        "redistributed."
    )
    assert manifest["redistribution_boundary"]["transcription"] == TRANSCRIPTION_SCOPE
    assert "nist" not in manifest_text.casefold()
    assert "source_rating" not in manifest_text


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("manifest.json", "SHA-256 mismatch for manifest.json"),
        (DATA_NAME, f"SHA-256 mismatch for {DATA_NAME}"),
    ],
)
def test_rejects_unsealed_bundle_tamper(
    tmp_path: Path, filename: str, message: str
) -> None:
    bundle_dir = _copy_bundle(tmp_path)
    path = bundle_dir / filename
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ExperimentalReferenceError, match=message):
        load_experimental_water_fundamentals(bundle_dir)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("3657.1", "3657.2", "unexpected wavenumber_cm1"),
        ("H2-16O", "H2O", "unexpected exact_isotopologue"),
        (",B2,", ",B1,", "unexpected symmetry_species"),
        ("cm^-1", "cm-1", "unexpected units"),
        (",gas,", ",liquid,", "unexpected phase"),
        (
            "dinu_2019_table1_h2_16o",
            "untracked_source",
            "unexpected source_id",
        ),
    ],
)
def test_rejects_resealed_csv_scientific_drift(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    bundle_dir = _copy_bundle(tmp_path)
    data_path = bundle_dir / DATA_NAME
    original = data_path.read_text(encoding="utf-8")
    assert old in original
    data_path.write_text(original.replace(old, new, 1), encoding="utf-8")
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    _seal_bundle(bundle_dir, manifest)

    with pytest.raises(ExperimentalReferenceError, match=message):
        load_experimental_water_fundamentals(bundle_dir)


def test_rejects_resealed_column_drift(tmp_path: Path) -> None:
    bundle_dir = _copy_bundle(tmp_path)
    data_path = bundle_dir / DATA_NAME
    original = data_path.read_text(encoding="utf-8")
    data_path.write_text(
        original.replace("wavenumber_cm1", "frequency_cm1", 1), encoding="utf-8"
    )
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    manifest["data"]["columns"][4] = "frequency_cm1"
    _seal_bundle(bundle_dir, manifest)

    with pytest.raises(
        ExperimentalReferenceError, match="manifest data inventory is inconsistent"
    ):
        load_experimental_water_fundamentals(bundle_dir)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (
            ("units", "wavenumber_cm1"),
            "m^-1",
            "wavenumber units must be cm",
        ),
        (
            ("value_definition", "phase"),
            "liquid",
            "observed gas-phase fundamentals",
        ),
        (
            ("sources", "dinu_2019_table1_h2_16o", "exact_isotopologue"),
            "H2O",
            "Table 1 sources or isotopologues",
        ),
        (
            (
                "underlying_citations",
                "toth_1999_d2o",
                "url",
            ),
            "https://example.invalid/d2o",
            "underlying Toth citations",
        ),
        (
            ("scope", "comparison_boundary"),
            "Validated against experiment.",
            "non-benchmark comparison scope",
        ),
        (
            ("scope", "intensities_included"),
            True,
            "intensities are absent",
        ),
        (
            ("redistribution_boundary", "not_redistributed"),
            "Spectrum included.",
            "article content and spectra are not redistributed",
        ),
    ],
)
def test_rejects_resealed_manifest_contract_drift(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: Any,
    message: str,
) -> None:
    bundle_dir = _copy_bundle(tmp_path)
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    _set_nested(manifest, path, replacement)
    _seal_bundle(bundle_dir, manifest)

    with pytest.raises(ExperimentalReferenceError, match=message):
        load_experimental_water_fundamentals(bundle_dir)
