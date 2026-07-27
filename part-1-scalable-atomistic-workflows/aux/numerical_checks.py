"""Mechanical tensor checks for learner-facing model comparisons.

This module deliberately knows nothing about Toolkit batches or model APIs.  A
notebook cell chooses the fields to hold fixed, calls the models, chooses the
outputs to compare, and applies every numerical acceptance limit.  These
helpers only make defensive tensor copies, check exact input preservation, and
reduce already-computed output differences.

The functions use the small tensor interface shared by PyTorch tensors and
NumPy arrays.  PyTorch values stay on their current device while they are
copied and compared; only the final scalar differences move to Python.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd


TensorMap = dict[str, Any]


def _selected_names(names: Iterable[str], *, argument: str) -> tuple[str, ...]:
    if isinstance(names, str):
        raise TypeError(f"{argument} must be an iterable of names, not a string")
    selected = tuple(names)
    if not selected:
        raise ValueError(f"{argument} must contain at least one name")
    if any(not isinstance(name, str) or not name for name in selected):
        raise ValueError(f"every entry in {argument} must be a non-empty string")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{argument} must not contain duplicate names")
    return selected


def _detached_clone(value: Any, *, label: str) -> Any:
    """Copy one tensor without retaining an autograd graph."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(value, "copy", None)
    if callable(copy):
        return copy()
    raise TypeError(f"{label} must be a tensor or array with clone() or copy()")


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError("tensor values must expose shape")
    return tuple(int(dimension) for dimension in shape)


def _metadata(value: Any) -> tuple[tuple[int, ...], str, str | None]:
    device = getattr(value, "device", None)
    return (
        _shape(value),
        str(getattr(value, "dtype", "unknown")),
        (None if device is None else str(device)),
    )


def _exactly_equal(left: Any, right: Any) -> bool:
    equal = getattr(left, "equal", None)
    if callable(equal):
        return bool(equal(right))
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _scalar(value: Any) -> float:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return float(value)


def _max_absolute_difference(left: Any, right: Any, *, label: str) -> float:
    left_shape = _shape(left)
    right_shape = _shape(right)
    if left_shape != right_shape:
        raise ValueError(f"{label} shapes differ: {left_shape} != {right_shape}")
    element_count = 1 if not left_shape else int(np.prod(left_shape, dtype=np.int64))
    if element_count == 0:
        raise ValueError(f"{label} must not be empty")

    # PyTorch performs this reduction on the tensor's current device.  The
    # double conversion avoids signed/unsigned integer subtraction issues when
    # this helper is used to describe a changed held field.
    double = getattr(left, "double", None)
    other_double = getattr(right, "double", None)
    if callable(double) and callable(other_double):
        difference = (double() - other_double()).abs().max()
        return _scalar(difference)
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(left_array - right_array)))


def _changed_element_count(left: Any, right: Any) -> int:
    not_equal = left != right
    count_nonzero = getattr(not_equal, "count_nonzero", None)
    if callable(count_nonzero):
        return int(_scalar(count_nonzero()))
    return int(np.count_nonzero(np.asarray(not_equal)))


def snapshot_tensor_fields(
    source: Any,
    *,
    field_names: Iterable[str],
) -> TensorMap:
    """Detach and clone the explicitly selected tensor attributes on ``source``.

    ``source`` may be a Toolkit ``Batch`` or any object with tensor-valued
    attributes.  The helper does not discover fields: the caller must state
    every field that should remain fixed.
    """

    selected = _selected_names(field_names, argument="field_names")
    snapshot: TensorMap = {}
    for name in selected:
        if not hasattr(source, name):
            raise AttributeError(f"source has no tensor field {name!r}")
        snapshot[name] = _detached_clone(
            getattr(source, name), label=f"source field {name!r}"
        )
    return snapshot


def assert_tensor_fields_unchanged(
    source: Any,
    snapshot: Mapping[str, Any],
    *,
    field_names: Iterable[str],
) -> None:
    """Assert exact preservation of the explicitly selected source fields.

    Shape, dtype, and device changes are reported separately from value
    changes.  A value-change message includes the number of changed elements
    and the maximum absolute difference; no numerical tolerance is applied.
    """

    selected = _selected_names(field_names, argument="field_names")
    for name in selected:
        if name not in snapshot:
            raise KeyError(f"snapshot has no entry for tensor field {name!r}")
        if not hasattr(source, name):
            raise AssertionError(f"held tensor field {name!r} is now missing")

        current = getattr(source, name)
        expected = snapshot[name]
        current_shape, current_dtype, current_device = _metadata(current)
        expected_shape, expected_dtype, expected_device = _metadata(expected)
        if current_shape != expected_shape:
            raise AssertionError(
                f"held tensor field {name!r} changed shape: "
                f"{expected_shape} -> {current_shape}"
            )
        if current_dtype != expected_dtype:
            raise AssertionError(
                f"held tensor field {name!r} changed dtype: "
                f"{expected_dtype} -> {current_dtype}"
            )
        if current_device != expected_device:
            raise AssertionError(
                f"held tensor field {name!r} changed device: "
                f"{expected_device} -> {current_device}"
            )
        if not _exactly_equal(current, expected):
            changed = _changed_element_count(current, expected)
            maximum = _max_absolute_difference(
                current, expected, label=f"held tensor field {name!r}"
            )
            raise AssertionError(
                f"held tensor field {name!r} changed values: "
                f"{changed} element(s), max absolute difference {maximum:.8g}"
            )


def clone_selected_outputs(
    outputs: Mapping[str, Any],
    *,
    output_names: Iterable[str],
) -> TensorMap:
    """Detach and clone selected model outputs before another model call."""

    selected = _selected_names(output_names, argument="output_names")
    cloned: TensorMap = {}
    for name in selected:
        if name not in outputs:
            raise KeyError(f"model outputs have no entry {name!r}")
        cloned[name] = _detached_clone(outputs[name], label=f"model output {name!r}")
    return cloned


def max_absolute_differences(
    left_outputs: Mapping[str, Any],
    right_outputs: Mapping[str, Any],
    *,
    output_names: Iterable[str],
) -> dict[str, float]:
    """Return per-output maximum absolute differences for selected outputs."""

    selected = _selected_names(output_names, argument="output_names")
    differences: dict[str, float] = {}
    for name in selected:
        if name not in left_outputs:
            raise KeyError(f"left outputs have no entry {name!r}")
        if name not in right_outputs:
            raise KeyError(f"right outputs have no entry {name!r}")
        differences[name] = _max_absolute_difference(
            left_outputs[name],
            right_outputs[name],
            label=f"model output {name!r}",
        )
    return differences


def build_difference_check_table(
    measurements: Mapping[str, Mapping[str, float]],
    limits: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Shape explicit measurements and limits into one learner-facing table.

    The caller chooses every comparison name, output name, and limit.  This
    helper only validates matching layouts and applies the strict ``value <
    limit`` relation used by the notebook; the caller still displays the table
    and decides whether a failed row should stop execution.
    """

    comparisons = _selected_names(measurements.keys(), argument="measurements")
    if set(comparisons) != set(limits):
        raise ValueError("measurements and limits must contain the same comparisons")

    rows: list[dict[str, str | float | bool]] = []
    for comparison in comparisons:
        observed = measurements[comparison]
        accepted = limits[comparison]
        outputs = _selected_names(observed.keys(), argument=f"{comparison} outputs")
        if set(outputs) != set(accepted):
            raise ValueError(
                f"measurements and limits for {comparison!r} must contain "
                "the same outputs"
            )
        for output in outputs:
            value = float(observed[output])
            limit = float(accepted[output])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"measurement {comparison}.{output} must be non-negative "
                    "and finite"
                )
            if not np.isfinite(limit) or limit <= 0.0:
                raise ValueError(
                    f"limit {comparison}.{output} must be positive and finite"
                )
            rows.append(
                {
                    "comparison": comparison,
                    "output": output,
                    "measured_max_abs_difference": value,
                    "strict_limit": limit,
                    "passed": value < limit,
                }
            )
    return pd.DataFrame(rows)


__all__ = (
    "assert_tensor_fields_unchanged",
    "build_difference_check_table",
    "clone_selected_outputs",
    "max_absolute_differences",
    "snapshot_tensor_fields",
)
