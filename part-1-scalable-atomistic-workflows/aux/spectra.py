"""Dependency-light spectrum analysis for predicted-charge trajectories."""

from __future__ import annotations

import numpy as np


E_ANGSTROM_TO_DEBYE = 4.803204712570263
SPEED_OF_LIGHT_CM_S = 2.99792458e10


def welch_dipole_current_spectrum(
    dipoles_e_angstrom: np.ndarray,
    dt_fs: float,
    *,
    segment_time_fs: float,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return wavenumber, mean PSD, and per-segment PSD.

    The current is the first difference of the *total* dipole, so it includes
    both ``q v`` and charge-flux ``r dq/dt`` terms. No quantum correction is
    applied. The PSD is a normalized classical predicted-charge IR proxy.
    Segment duration and overlap are required methodology inputs rather than
    hidden signal-processing defaults.
    """

    dipoles = np.asarray(dipoles_e_angstrom, dtype=np.float64)
    if dipoles.ndim != 2 or dipoles.shape[1] != 3:
        raise ValueError("dipoles must have shape (frames, 3)")
    current = np.diff(dipoles * E_ANGSTROM_TO_DEBYE, axis=0) / float(dt_fs)
    nperseg = int(round(segment_time_fs / float(dt_fs)))
    if nperseg < 8:
        raise ValueError("segment_time_fs is too short")
    if current.shape[0] < nperseg:
        raise ValueError(
            f"Need at least {nperseg + 1} dipole frames for a "
            f"{segment_time_fs:g} fs Welch segment"
        )
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, int(round(nperseg * (1.0 - overlap))))
    starts = range(0, current.shape[0] - nperseg + 1, step)
    window = np.hanning(nperseg)
    window_power = np.sum(window**2)

    segment_psd = []
    for start in starts:
        segment = current[start : start + nperseg]
        segment = segment - segment.mean(axis=0, keepdims=True)
        transform = np.fft.rfft(segment * window[:, None], axis=0)
        power = (transform.real**2 + transform.imag**2).sum(axis=1)
        power /= 3.0 * window_power
        segment_psd.append(power)
    per_segment = np.asarray(segment_psd)
    mean_psd = per_segment.mean(axis=0)
    frequencies_hz = np.fft.rfftfreq(nperseg, d=float(dt_fs) * 1e-15)
    wavenumber_cm1 = frequencies_hz / SPEED_OF_LIGHT_CM_S
    return wavenumber_cm1, mean_psd, per_segment


def band_centroid(
    wavenumber_cm1: np.ndarray,
    intensity: np.ndarray,
    window_cm1: tuple[float, float],
) -> float:
    """Return the area-weighted centroid in a predeclared spectral window."""

    x = np.asarray(wavenumber_cm1, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    lo, hi = window_cm1
    mask = (x >= lo) & (x <= hi)
    if mask.sum() < 2:
        raise ValueError("Centroid window contains fewer than two points")
    x_window = x[mask]
    y_window = y[mask]
    dx = np.diff(x_window)
    area = np.sum(0.5 * (y_window[1:] + y_window[:-1]) * dx)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("Centroid window has no positive spectral area")
    xy = x_window * y_window
    moment = np.sum(0.5 * (xy[1:] + xy[:-1]) * dx)
    return float(moment / area)


__all__ = [
    "E_ANGSTROM_TO_DEBYE",
    "SPEED_OF_LIGHT_CM_S",
    "band_centroid",
    "welch_dipole_current_spectrum",
]
