#!/usr/bin/env python3
"""Write numeric geometry diagnostics for the adsorption tutorial panel."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from helpers import (
    build_alpha_alumina_0001_slab,
    build_config_grid,
    build_cu111_slab,
    build_pd111_slab,
    sites_for_host,
)


def _z_layers(atoms, tol: float = 0.35) -> list[list[float]]:
    layers: list[list[float]] = []
    for z in sorted(float(v) for v in atoms.positions[:, 2]):
        if not layers or abs(z - layers[-1][-1]) > tol:
            layers.append([z])
        else:
            layers[-1].append(z)
    return layers


def _nearest_ads_slab_distance(config, slab_n: int) -> float:
    slab = config.atoms[:slab_n]
    ads = config.atoms[slab_n:]
    distances = np.linalg.norm(
        ads.positions[:, None, :] - slab.positions[None, :, :],
        axis=2,
    )
    return float(distances.min())


def main() -> int:
    out_dir = Path("part-1-batched-adsorption/outputs/geometry_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    hosts = {
        "Cu(111)": build_cu111_slab(
            min_slab_size=8.0,
            min_vacuum_size=15.0,
            supercell=(3, 3, 1),
        ),
        "Pd(111)": build_pd111_slab(
            min_slab_size=8.0,
            min_vacuum_size=15.0,
            supercell=(3, 3, 1),
        ),
        "Al2O3(0001)": build_alpha_alumina_0001_slab(
            min_slab_size=8.0,
            min_vacuum_size=15.0,
            supercell=(2, 2, 1),
        ),
    }

    slab_rows = []
    site_rows = []
    for host, atoms in hosts.items():
        layers = _z_layers(atoms)
        z_span = float(atoms.positions[:, 2].max() - atoms.positions[:, 2].min())
        cell_lengths = atoms.cell.lengths()
        slab_rows.append(
            {
                "host": host,
                "atoms": len(atoms),
                "cell_a_A": cell_lengths[0],
                "cell_b_A": cell_lengths[1],
                "cell_c_A": cell_lengths[2],
                "slab_thickness_A": z_span,
                "vacuum_gap_A": float(cell_lengths[2] - z_span),
                "n_z_layers": len(layers),
                "layer_counts": [len(layer) for layer in layers],
                "pbc": atoms.pbc.tolist(),
            }
        )
        for site_name, positions in sites_for_host(host, atoms).items():
            for idx, position in enumerate(positions):
                site_rows.append(
                    {
                        "host": host,
                        "site": site_name,
                        "site_index": idx,
                        "x_A": float(position[0]),
                        "y_A": float(position[1]),
                        "z_A": float(position[2]),
                    }
                )

    placement_rows = []
    for host in ("Cu(111)", "Pd(111)"):
        slab = hosts[host]
        configs = build_config_grid(
            host_name=host,
            slab=slab,
            adsorbate_name="CO",
            sites_filter=["top", "bridge", "fcc", "hcp"],
            orientations_filter=["C-down", "O-down"],
            rotations_deg=(0.0, 60.0),
            heights_A=(1.6, 1.8, 2.0, 2.2, 2.4),
        )
        for config in configs:
            placement_rows.append(
                {
                    "host": host,
                    "adsorbate": "CO",
                    "label": config.label,
                    "site": config.site,
                    "orientation": config.orientation,
                    "rot_deg": config.rot_deg,
                    "height_A": config.height,
                    "nearest_ads_slab_distance_A": _nearest_ads_slab_distance(
                        config,
                        len(slab),
                    ),
                }
            )

    pd.DataFrame(slab_rows).to_csv(out_dir / "slab_diagnostics.csv", index=False)
    pd.DataFrame(site_rows).to_csv(out_dir / "site_coordinates.csv", index=False)
    pd.DataFrame(placement_rows).to_csv(
        out_dir / "co_initial_placement_distances.csv",
        index=False,
    )
    with open(out_dir / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "slab_diagnostics": str(out_dir / "slab_diagnostics.csv"),
                "site_coordinates": str(out_dir / "site_coordinates.csv"),
                "co_initial_placement_distances": str(
                    out_dir / "co_initial_placement_distances.csv"
                ),
            },
            f,
            indent=2,
        )
    print(f"Wrote geometry audit files under {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
