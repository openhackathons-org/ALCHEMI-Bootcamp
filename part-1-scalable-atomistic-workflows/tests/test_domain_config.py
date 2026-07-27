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

    assert record["schema"] == DOMAIN_METHODOLOGY_SCHEMA
    assert record["name"] == "part1-packmol-domain-decomposition"
    assert record["version"] == "1.6.0"
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
    counts = config.capacity_molecules_per_species

    assert config.nci_system_id.strip()
    assert config.nci_scale > 0
    assert config.atoms_per_composition_unit > 0
    assert config.aimnet_neighbor_cutoff_a > 0
    assert all(right == 2 * left for left, right in zip(counts, counts[1:]))
    assert config.live_molecules_per_species in counts
    assert config.electrostatics_validation_molecules_per_species in counts
    assert config.parity_molecules_per_species in counts
    assert set(config.capacity_world_sizes) <= set(config.supported_world_sizes)
    assert all(value > 1 for value in config.distributed_world_sizes)
    assert config.steady_timing_world_sizes == config.supported_world_sizes
    assert config.steady_timing_warmup_count >= 1
    assert config.steady_timing_sample_count >= 5
    assert config.steady_timing_max_relative_iqr == 0.10
    assert config.steady_timing_model_evaluations_per_workflow == 2
    assert config.domain_parallel_multi_rank_initial_force_evaluations == 1
    assert config.steady_timing_run_steps(1) == 2
    assert all(
        config.steady_timing_run_steps(world_size) == 1
        for world_size in config.distributed_world_sizes
    )
    assert config.domain_grid_dims is None
    assert config.campaign_world_sizes == (1, 2, 4)
    assert config.supported_world_sizes == config.campaign_world_sizes
    assert config.force_reference_world_size == 1
    assert config.force_comparison_world_sizes == (2, 4)
    assert config.energy_reference_world_size == 2
    assert config.energy_comparison_world_sizes == (4,)

    unsupported = max(config.supported_world_sizes) + 1
    with pytest.raises(ValueError, match="unsupported steady-timing"):
        config.steady_timing_run_steps(unsupported)


def test_explicit_zero_d3_smoothing_is_valid() -> None:
    config = replace(DOMAIN_METHODOLOGY, d3_smoothing_fraction=0.0)

    assert config.d3_smoothing_fraction == 0.0
    assert config.resolved_values()["d3_smoothing_fraction"] == 0.0
    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        replace(DOMAIN_METHODOLOGY, d3_smoothing_fraction=1.0)


def test_steady_timing_keeps_at_least_one_requested_multi_rank_step() -> None:
    with pytest.raises(ValueError, match="at least one requested multi-rank step"):
        replace(
            DOMAIN_METHODOLOGY,
            steady_timing_model_evaluations_per_workflow=1,
        )


def test_steady_timing_relative_iqr_limit_is_a_fraction() -> None:
    with pytest.raises(ValueError, match="must not exceed 1"):
        replace(DOMAIN_METHODOLOGY, steady_timing_max_relative_iqr=1.01)
    with pytest.raises(ValueError, match="positive and finite"):
        replace(DOMAIN_METHODOLOGY, steady_timing_max_relative_iqr=0.0)
