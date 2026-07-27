"""Checks for the shared isotope and trajectory methodology values."""

from __future__ import annotations

from pathlib import Path
import sys


PART_DIR = Path(__file__).resolve().parents[1]
if str(PART_DIR) not in sys.path:
    sys.path.insert(0, str(PART_DIR))

from aux import workflow_config as config  # noqa: E402


def test_workflow_config_exports_every_public_setting() -> None:
    exported = set(config.__all__)
    public = {
        name
        for name in vars(config)
        if name.isupper() and not name.startswith("_")
    }

    assert exported == public


def test_statuses_and_checks_are_well_formed() -> None:
    assert config.IR_WARMUP_STATUS == 0
    assert config.IR_PRODUCTION_STATUS == 1
    assert config.IR_WARMUP_STATUS != config.IR_PRODUCTION_STATUS

    float_settings = [
        value
        for name in config.__all__
        if isinstance((value := getattr(config, name)), float)
    ]
    assert float_settings
    assert all(value > 0.0 for value in float_settings)


def test_main_ir_methodology_matches_the_recorded_workflow() -> None:
    assert config.IR_FIRE_INITIAL_DT == 0.01
    assert config.IR_NVT_FRICTION_PER_FS == 0.01
    assert config.IR_INITIAL_VELOCITY_RANDOM_SEEDS == (101, 101, 202, 202)
    assert config.IR_NVT_RANDOM_SEED == 303

    assert config.IR_WELCH_SEGMENT_TIME_FS == 5_000.0
    assert config.IR_WELCH_OVERLAP_FRACTION == 0.5
    assert 0.0 <= config.IR_WELCH_OVERLAP_FRACTION < 1.0
    assert all(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
        for seed in config.IR_INITIAL_VELOCITY_RANDOM_SEEDS
    )


def test_equal_current_values_keep_separate_semantic_names() -> None:
    assert (
        config.IR_CAPTURE_CHARGE_TOLERANCE_E
        == config.IR_CHARGE_NEUTRALITY_TOLERANCE_E
    )
    assert (
        config.IR_CHARGE_NEUTRALITY_TOLERANCE_E
        == config.IR_DIPOLE_ORIGIN_TOLERANCE_E_ANGSTROM
    )
