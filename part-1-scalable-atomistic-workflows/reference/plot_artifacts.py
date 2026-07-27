#!/usr/bin/env python3
"""Plot the checksummed B97-3c H/D reference bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.reference import (  # noqa: E402
    hann_resolution_envelope,
    load_psi4_b973c_ir_artifact,
    match_isotopologue_modes,
    reference_water_monomer_mode_labels,
    reference_water_ring_mode_characters,
)
from aux.plotting import (  # noqa: E402
    FIGURE_SIZE,
    SYSTEM_DISPLAY_LABELS,
    style_axis,
)


COLORS = {
    "bend": "#7B2CBF",
    "symmetric_stretch": "#2A9D8F",
    "antisymmetric_stretch": "#0077B6",
    "hbonded_oh": "#D55E00",
    "free_oh": "#009E73",
    "intermolecular": "#7A7A7A",
}

def target_characters(source_characters, mode_map) -> tuple[str, ...]:
    labels = [""] * len(source_characters)
    for source_mode, target_mode in enumerate(mode_map.source_to_target):
        labels[int(target_mode)] = source_characters[source_mode]
    return tuple(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "b97_3c_ir_reference.png",
    )
    args = parser.parse_args()

    directories = {"H2O": "h2o", "D2O": "d2o", "(H2O)6": "h6", "(D2O)6": "d6"}
    references = {
        label: load_psi4_b973c_ir_artifact(args.artifact_root / directory)
        for label, directory in directories.items()
    }

    monomer_h = reference_water_monomer_mode_labels(references["H2O"])
    monomer_map = match_isotopologue_modes(references["H2O"], references["D2O"])
    ring_h = reference_water_ring_mode_characters(references["(H2O)6"]).dominant_labels
    ring_map = match_isotopologue_modes(references["(H2O)6"], references["(D2O)6"])
    characters = {
        "H2O": monomer_h,
        "D2O": target_characters(monomer_h, monomer_map),
        "(H2O)6": ring_h,
        "(D2O)6": target_characters(ring_h, ring_map),
    }

    grid = np.linspace(100.0, 4200.0, 6000)
    figure, axes = plt.subplots(
        2, 2, figsize=FIGURE_SIZE, sharex=True, sharey=True
    )
    legend_handles: dict[str, object] = {}
    for axis, label in zip(axes.flat, directories, strict=True):
        reference = references[label]
        positive = reference.frequencies_cm1 > 0.0
        frequencies = reference.frequencies_cm1[positive]
        intensities = reference.ir_intensities_km_mol[positive]
        labels = np.asarray(characters[label], dtype=object)[positive]
        envelope = hann_resolution_envelope(
            frequencies,
            intensities,
            grid,
            dt_fs=0.5,
            segment_time_fs=5000.0,
        )
        # Reference arrays are intentionally immutable after checksum
        # validation; use an out-of-place normalization for plotting.
        envelope = envelope / envelope.max()
        envelope_line = axis.plot(
            grid,
            envelope,
            color="#202020",
            linewidth=1.15,
            label="5 ps Hann response",
        )[0]
        legend_handles.setdefault("5 ps Hann response", envelope_line)
        intensity_scale = intensities.max()
        for frequency, intensity, character in zip(
            frequencies, intensities, labels, strict=True
        ):
            stick = axis.vlines(
                frequency,
                0.0,
                intensity / intensity_scale,
                color=COLORS[str(character)],
                linewidth=1.0,
                alpha=0.9,
            )
            legend_handles.setdefault(str(character), stick)
        axis.set_title(SYSTEM_DISPLAY_LABELS[label])
        axis.set_xlim(100.0, 4200.0)
        axis.set_ylim(0.0, 1.05)
        style_axis(axis, grid_axis="x")

    for axis in axes[:, 0]:
        axis.set_ylabel("independently normalized intensity")
    for axis in axes[-1]:
        axis.set_xlabel("wavenumber / cm$^{-1}$")
    figure.legend(
        legend_handles.values(),
        [label.replace("_", " ") for label in legend_handles],
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
    )
    figure.suptitle("B97-3c double-harmonic IR: raw sticks and finite-window response")
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
