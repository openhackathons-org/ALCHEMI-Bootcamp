# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

_CORE_DIR = Path(__file__).resolve().parents[1]
_RECORD_DIR = _CORE_DIR / "data" / "science_context"


def _record(name: str) -> dict:
    return json.loads((_RECORD_DIR / name).read_text(encoding="utf-8"))


def test_model_scope_records_exact_aliases_and_composition() -> None:
    payload = _record("model_scope.json")
    models = {model["name"]: model for model in payload["models"]}

    mace = models["MACE-MPA-0"]
    assert mace["checkpoint_alias"] == "medium-mpa-0"
    assert mace["elements_covered"] == 89
    assert mace["training_datasets"] == ["MPTrj", "sAlex"]
    assert mace["reference_level"] == "DFT (PBE+U)"
    assert mace["declared_target"] == "Materials"

    aimnet = models["AIMNet2 wB97M-D3"]
    assert len(aimnet["supported_elements"]) == 14
    assert aimnet["electronic_structure_scope"] == "Closed-shell"
    assert "learned local energy" in aimnet["course_wording"]
    assert "electrostatics that read the predicted charges" in aimnet["course_wording"]


def test_official_documentation_map_contains_resolved_https_links() -> None:
    payload = _record("official_docs.json")
    assert payload["documentation_version_observed"] == "0.2.0"
    assert len(payload["links"]) == 6
    assert all(link["status"] == "resolved" for link in payload["links"])
    assert all(link["url"].startswith("https://") for link in payload["links"])
    assert len({link["topic"] for link in payload["links"]}) == 6


def test_nci_baseline_source_hash_and_curve_grid() -> None:
    payload = _record("nci_baseline.json")
    source = payload["source_data"]
    data_file = _CORE_DIR.parents[1] / source["local_file"]

    assert hashlib.sha256(data_file.read_bytes()).hexdigest() == source["sha256"]
    assert source["license_audit"]["release_blocker"] is True
    assert source["license_audit"]["upstream_version"] is None

    curves = payload["curves"]
    assert len(curves) == 30
    assert len({(row["system_id"], row["scale"]) for row in curves}) == 30
    assert {row["system_name"] for row in curves} == {
        "phenol - N-methylacetamide",
        "propyne - methyl azide",
        "ammonia - benzoate",
    }
    assert all(math.isfinite(value) for row in curves for value in row.values() if isinstance(value, float))


def test_nci_equilibrium_component_shifts_match_summary() -> None:
    payload = _record("nci_baseline.json")
    equilibrium = {
        row["system_name"]: row for row in payload["curves"] if row["scale"] == 1.0
    }

    for summary in payload["summary"]:
        row = equilibrium[summary["system"]]
        d3_shift = row["local_plus_d3_kcal_mol"] - row["aimnet_local_kcal_mol"]
        electrostatic_shift = (
            row["complete_kcal_mol"] - row["local_plus_d3_kcal_mol"]
        )
        assert d3_shift == pytest.approx(
            summary["d3_shift_at_equilibrium_kcal_mol"], abs=2e-6
        )
        assert electrostatic_shift == pytest.approx(
            summary["electrostatic_shift_at_equilibrium_kcal_mol"], abs=2e-6
        )


def test_nci_comparison_tolerance_covers_observed_cuda_repeat_delta() -> None:
    payload = _record("nci_baseline.json")
    tolerance = payload["comparison_tolerance_kcal_mol"]
    observed = payload["repeatability_check"][
        "maximum_absolute_repeat_delta_kcal_mol"
    ]

    assert max(observed.values()) <= tolerance
    assert tolerance == pytest.approx(0.05)
