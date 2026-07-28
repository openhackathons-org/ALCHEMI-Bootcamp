"""Checks for the one-source Part 1 domain settings."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import sys

import pytest

PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.domain.config import (  # noqa: E402
    DOMAIN_METHODOLOGY,
    DOMAIN_METHODOLOGY_SCHEMA,
    DomainMethodologyConfig,
)


def test_every_domain_setting_has_required_context() -> None:
    record = DOMAIN_METHODOLOGY.as_record()

    assert DOMAIN_METHODOLOGY_SCHEMA == "alchemi.part1-domain-methodology.v5"
    assert record["schema"] == DOMAIN_METHODOLOGY_SCHEMA
    assert record["name"] == "part1-packmol-domain-decomposition"
    assert record["version"] == "1.10.0"
    assert set(record["settings"]) == {
        item.name
        for item in fields(DomainMethodologyConfig)
        if item.name not in {"name", "version"}
    }
    for name, setting in record["settings"].items():
        assert setting["name"] == name
        assert setting["units"]
        assert setting["scope"]
        assert setting["rationale"]
        assert setting["source"]


def test_domain_settings_form_one_consistent_campaign() -> None:
    config = DOMAIN_METHODOLOGY

    assert config.nci_system_id.strip()
    assert config.nci_scale > 0
    assert config.atoms_per_composition_unit > 0
    assert config.aimnet_neighbor_cutoff_a > 0
    assert config.live_molecules_per_species == 128
    assert config.electrostatics_validation_molecules_per_species == 128
    assert config.fixed_molecules_per_species == 2_048
    assert (
        config.fixed_molecules_per_species * config.atoms_per_composition_unit == 51_200
    )
    assert all(value > 1 for value in config.distributed_world_sizes)
    assert config.evaluation_warmup_count == 1
    assert config.evaluation_pass_count == 3
    assert config.measured_model_evaluations_per_pass == 1
    assert config.domain_parallel_multi_rank_warmup_force_prime_evaluations == 1
    assert config.domain_grid_dims is None
    assert config.campaign_world_sizes == (1, 2, 4)
    assert config.supported_world_sizes == config.campaign_world_sizes
    assert config.force_reference_world_size == 1
    assert config.force_comparison_world_sizes == (2, 4)
    assert config.energy_reference_world_size == 2
    assert config.energy_comparison_world_sizes == (4,)
    assert config.evaluation_energy_dtype_for_world_size(1) == "torch.float32"
    assert config.evaluation_energy_dtype_for_world_size(2) == "torch.float64"
    assert config.evaluation_energy_dtype_for_world_size(4) == "torch.float64"


@pytest.mark.parametrize("world_size", (True, 1.0, 3))
def test_energy_dtype_mapping_rejects_unsupported_world_sizes(
    world_size: object,
) -> None:
    with pytest.raises(ValueError, match="world_size must be one of"):
        DOMAIN_METHODOLOGY.evaluation_energy_dtype_for_world_size(world_size)  # type: ignore[arg-type]


def test_energy_dtype_mapping_rejects_changed_dtypes() -> None:
    with pytest.raises(
        ValueError,
        match="evaluation_energy_dtype_single_rank must be torch.float32",
    ):
        replace(
            DOMAIN_METHODOLOGY,
            evaluation_energy_dtype_single_rank="torch.float64",
        )

    with pytest.raises(
        ValueError,
        match="evaluation_energy_dtype_multi_rank must be torch.float64",
    ):
        replace(
            DOMAIN_METHODOLOGY,
            evaluation_energy_dtype_multi_rank="torch.float32",
        )


def test_fixed_evaluation_position_tolerance_is_declared_and_positive() -> None:
    assert DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a == 1.0e-4
    assert (
        DOMAIN_METHODOLOGY.resolved_values()["evaluation_position_mic_tolerance_a"]
        == 1.0e-4
    )
    with pytest.raises(
        ValueError,
        match="evaluation_position_mic_tolerance_a must be positive and finite",
    ):
        replace(
            DOMAIN_METHODOLOGY,
            evaluation_position_mic_tolerance_a=0.0,
        )


def test_distributed_energy_tolerances_are_declared_separately() -> None:
    repeatability_name = "distributed_energy_repeatability_tolerance_ev_per_atom"
    assert (
        DOMAIN_METHODOLOGY.distributed_energy_repeatability_tolerance_ev_per_atom
        == 1.0e-4
    )
    assert DOMAIN_METHODOLOGY.evaluation_energy_tolerance_ev_per_atom == 1.0e-4
    resolved = DOMAIN_METHODOLOGY.resolved_values()
    assert resolved[repeatability_name] == 1.0e-4
    assert resolved["evaluation_energy_tolerance_ev_per_atom"] == 1.0e-4

    with pytest.raises(
        ValueError,
        match=(
            "distributed_energy_repeatability_tolerance_ev_per_atom "
            "must be positive and finite"
        ),
    ):
        replace(
            DOMAIN_METHODOLOGY,
            distributed_energy_repeatability_tolerance_ev_per_atom=0.0,
        )


def test_explicit_zero_d3_smoothing_is_valid() -> None:
    config = replace(DOMAIN_METHODOLOGY, d3_smoothing_fraction=0.0)

    assert config.d3_smoothing_fraction == 0.0
    assert config.resolved_values()["d3_smoothing_fraction"] == 0.0
    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        replace(DOMAIN_METHODOLOGY, d3_smoothing_fraction=1.0)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("evaluation_warmup_count", 2, "exactly 1"),
        ("evaluation_pass_count", 2, "exactly 3"),
        ("measured_model_evaluations_per_pass", 2, "exactly 1"),
    ),
)
def test_fixed_evaluation_counts_cannot_drift(
    field_name: str,
    value: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(
            DOMAIN_METHODOLOGY,
            **{field_name: value},
        )
