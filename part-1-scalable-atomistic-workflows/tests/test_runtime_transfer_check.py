"""Checks for the stock Toolkit buffer-copy preflight used by the campaign."""

from __future__ import annotations

from pathlib import Path
import sys


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.runtime import check_batch_buffer_transfer  # noqa: E402


def test_pinned_stock_core_reports_remaining_full_dtype_failures_on_cpu() -> None:
    report = check_batch_buffer_transfer("cpu")

    assert report["device"] == "cpu"
    assert report["passed"] is False
    assert [case["float_dtype"] for case in report["cases"]] == [
        "float32",
        "float64",
    ]
    for case in report["cases"]:
        assert case["passed"] is False
        assert set(case["checks"]) == {
            "first_put",
            "repeated_put",
            "zero_then_put",
        }

        first_put = case["checks"]["first_put"]
        assert first_put["passed"] is False
        assert "atomic_numbers" in first_put["mismatches"]
        assert "atom_code" in first_put["mismatches"]

        # Exact secondary mismatches can vary with Torch and Warp versions.
        # Keep their result shape visible without baking one runtime's full
        # mismatch list into this test.
        for check in case["checks"].values():
            assert isinstance(check["passed"], bool)
            assert isinstance(check["mismatches"], list)

        assert all(
            mismatch.startswith(
                ("first_put.", "repeated_put.", "zero_then_put.")
            )
            for mismatch in case["mismatches"]
        )
