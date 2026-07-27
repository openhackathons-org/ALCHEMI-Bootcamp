"""Focused tests for the pure tensor-check helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.numerical_checks import (  # noqa: E402
    assert_tensor_fields_unchanged,
    build_difference_check_table,
    clone_selected_outputs,
    max_absolute_differences,
    snapshot_tensor_fields,
)


def test_snapshot_clones_only_explicitly_selected_fields() -> None:
    source = SimpleNamespace(
        positions=np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
        atomic_numbers=np.array([8], dtype=np.int64),
        ignored=np.array([99.0]),
    )

    snapshot = snapshot_tensor_fields(
        source, field_names=("positions", "atomic_numbers")
    )
    source.positions[0, 0] = 4.0

    assert tuple(snapshot) == ("positions", "atomic_numbers")
    assert snapshot["positions"][0, 0] == 0.0
    assert "ignored" not in snapshot


def test_unchanged_check_accepts_exact_fields() -> None:
    source = SimpleNamespace(
        positions=np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
        neighbor_matrix=np.array([[0, 1], [1, 0]], dtype=np.int64),
    )
    names = ("positions", "neighbor_matrix")
    snapshot = snapshot_tensor_fields(source, field_names=names)

    assert_tensor_fields_unchanged(source, snapshot, field_names=names)


def test_unchanged_check_reports_field_count_and_maximum() -> None:
    source = SimpleNamespace(
        positions=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    )
    snapshot = snapshot_tensor_fields(source, field_names=("positions",))
    source.positions[0, 1] = 1.5
    source.positions[1, 0] = -1.0

    with pytest.raises(AssertionError) as error:
        assert_tensor_fields_unchanged(source, snapshot, field_names=("positions",))

    message = str(error.value)
    assert "positions" in message
    assert "2 element(s)" in message
    assert "max absolute difference 3" in message


def test_unchanged_check_reports_metadata_changes_separately() -> None:
    source = SimpleNamespace(values=np.array([1.0, 2.0], dtype=np.float32))
    snapshot = snapshot_tensor_fields(source, field_names=("values",))
    source.values = source.values.astype(np.float64)

    with pytest.raises(AssertionError, match="changed dtype: float32 -> float64"):
        assert_tensor_fields_unchanged(source, snapshot, field_names=("values",))


def test_unchanged_check_requires_a_snapshot_for_each_requested_field() -> None:
    source = SimpleNamespace(values=np.array([1.0]))

    with pytest.raises(KeyError, match="'values'"):
        assert_tensor_fields_unchanged(source, {}, field_names=("values",))


def test_output_clone_is_independent_of_reused_output_storage() -> None:
    outputs = {
        "energy": np.array([[1.0]], dtype=np.float32),
        "forces": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "ignored": np.array([5.0]),
    }

    cloned = clone_selected_outputs(outputs, output_names=("energy", "forces"))
    outputs["energy"][0, 0] = 9.0
    outputs["forces"].fill(0.0)

    np.testing.assert_array_equal(cloned["energy"], [[1.0]])
    np.testing.assert_array_equal(
        cloned["forces"], np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    )
    assert "ignored" not in cloned


def test_max_absolute_differences_follow_requested_output_order() -> None:
    eager = {
        "energy": np.array([[1.0], [4.0]]),
        "forces": np.array([[1.0, -2.0, 0.0]]),
        "charges": np.array([0.25, -0.25]),
    }
    compiled = {
        "energy": np.array([[1.5], [3.0]]),
        "forces": np.array([[1.0, 1.0, -0.5]]),
        "charges": np.array([0.20, -0.20]),
    }

    differences = max_absolute_differences(
        compiled,
        eager,
        output_names=("forces", "energy", "charges"),
    )

    assert tuple(differences) == ("forces", "energy", "charges")
    assert differences == pytest.approx({"forces": 3.0, "energy": 1.0, "charges": 0.05})


def test_max_absolute_differences_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="model output 'energy' shapes differ"):
        max_absolute_differences(
            {"energy": np.zeros((2, 1))},
            {"energy": np.zeros(2)},
            output_names=("energy",),
        )


def test_max_absolute_differences_reject_empty_outputs() -> None:
    with pytest.raises(ValueError, match="model output 'forces' must not be empty"):
        max_absolute_differences(
            {"forces": np.empty((0, 3))},
            {"forces": np.empty((0, 3))},
            output_names=("forces",),
        )


def test_difference_check_table_preserves_explicit_order_and_strict_limits() -> None:
    table = build_difference_check_table(
        {
            "compiled - eager": {"energy": 1.0e-6, "forces": 5.0e-6},
            "compiled repeat": {"energy": 0.0, "forces": 2.0e-6},
        },
        {
            "compiled - eager": {"energy": 5.0e-6, "forces": 5.0e-6},
            "compiled repeat": {"energy": 2.0e-6, "forces": 2.0e-6},
        },
    )

    assert table[["comparison", "output"]].values.tolist() == [
        ["compiled - eager", "energy"],
        ["compiled - eager", "forces"],
        ["compiled repeat", "energy"],
        ["compiled repeat", "forces"],
    ]
    assert table["passed"].tolist() == [True, False, True, False]


def test_difference_check_table_requires_matching_layouts() -> None:
    with pytest.raises(ValueError, match="same outputs"):
        build_difference_check_table(
            {"route": {"energy": 0.0}},
            {"route": {"forces": 1.0}},
        )


@pytest.mark.parametrize(
    ("function", "argument"),
    [
        (
            lambda: snapshot_tensor_fields(
                SimpleNamespace(value=np.array([1.0])), field_names="value"
            ),
            "field_names",
        ),
        (
            lambda: clone_selected_outputs(
                {"energy": np.array([1.0])}, output_names=("energy", "energy")
            ),
            "output_names",
        ),
    ],
)
def test_selected_names_must_be_explicit_and_unique(function, argument: str) -> None:
    with pytest.raises((TypeError, ValueError), match=argument):
        function()
