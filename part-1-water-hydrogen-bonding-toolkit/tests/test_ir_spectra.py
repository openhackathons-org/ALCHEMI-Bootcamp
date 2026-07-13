"""Tests for dependency-light trajectory spectrum analysis."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.spectra import band_centroid, welch_dipole_current_spectrum


def test_welch_returns_expected_grid_and_segment_count() -> None:
    dt_fs = 0.5
    frames = 41
    time_fs = np.arange(frames) * dt_fs
    dipoles = np.column_stack(
        (
            np.sin(2.0 * np.pi * time_fs / 5.0),
            np.zeros(frames),
            np.zeros(frames),
        )
    )

    wavenumber, mean_psd, per_segment = welch_dipole_current_spectrum(
        dipoles,
        dt_fs,
        segment_time_fs=10.0,
        overlap=0.5,
    )

    assert wavenumber.shape == mean_psd.shape == (11,)
    assert per_segment.shape == (3, 11)
    np.testing.assert_allclose(mean_psd, per_segment.mean(axis=0))
    assert np.all(np.diff(wavenumber) > 0.0)


def test_welch_rejects_insufficient_frames() -> None:
    with pytest.raises(ValueError, match="Need at least 21 dipole frames"):
        welch_dipole_current_spectrum(
            np.zeros((20, 3)),
            0.5,
            segment_time_fs=10.0,
        )


def test_band_centroid_integrates_a_predeclared_window() -> None:
    wavenumber = np.array([1000.0, 1100.0, 1200.0, 1300.0])
    intensity = np.array([0.0, 1.0, 1.0, 0.0])

    assert band_centroid(wavenumber, intensity, (1000.0, 1300.0)) == pytest.approx(
        1150.0
    )


def test_band_centroid_rejects_zero_area() -> None:
    with pytest.raises(ValueError, match="no positive spectral area"):
        band_centroid(
            np.array([1000.0, 1100.0]),
            np.zeros(2),
            (1000.0, 1100.0),
        )
