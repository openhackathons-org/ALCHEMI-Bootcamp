#!/usr/bin/env python3
"""Build the small NCI Atlas subset used by the Part 3 research notebook.

The source checkout is not modified. The output is a deterministic gzip CSV
containing ten separation points and frozen AB/A/B records for three systems.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import re
from pathlib import Path


SOURCE_REVISION = "1816bfc72609d7deb1d4f93ab9e27eb13bb44bec"
SELECTIONS = (
    {
        "subset": "HB375x10",
        "system_id": "1.041",
        "system_name": "phenol - N-methylacetamide",
        "interaction_class": "neutral hydrogen bond",
        "gradient_file": "hb375x10.xyz",
        "geometry_dir": "NCIA_HB375x10",
    },
    {
        "subset": "D442x10",
        "system_id": "1.07.74",
        "system_name": "propyne - methyl azide",
        "interaction_class": "dispersion-dominated",
        "gradient_file": "d442x10.xyz",
        "geometry_dir": "NCIA_D442x10",
    },
    {
        "subset": "IHB100x10",
        "system_id": "08.007",
        "system_name": "ammonia - benzoate",
        "interaction_class": "ionic hydrogen bond",
        "gradient_file": "ihb100x10.xyz",
        "geometry_dir": "NCIA_IHB100x10",
    },
)

FIELDNAMES = (
    "subset",
    "system_id",
    "system_name",
    "interaction_class",
    "scale",
    "fragment",
    "charge",
    "natoms",
    "symbols",
    "positions_angstrom",
    "wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol",
    "ccsd_t_cbs_interaction_energy_kcal_mol",
    "source_gradient_block",
    "source_geometry_file",
)


def parse_header_value(header: str, key: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(key)}=([^\s]+)", header)
    if match is None:
        raise ValueError(f"Missing {key!r} in header: {header}")
    return match.group(1)


def read_extended_xyz(path: Path, wanted_id: str) -> dict[float, dict[str, dict]]:
    records: dict[float, dict[str, dict]] = {}
    with path.open(encoding="utf-8") as handle:
        while name_line := handle.readline():
            name = name_line.strip()
            if not name:
                continue
            natoms = int(handle.readline())
            header = handle.readline().strip()
            atom_rows = [handle.readline().split() for _ in range(natoms)]
            if not name.startswith(f"{wanted_id}_"):
                continue

            name_match = re.match(r"^(.*)_(ab|a|b)$", name)
            if name_match is None:
                raise ValueError(f"Unrecognized fragment name: {name}")
            stem, fragment = name_match.groups()
            scale_matches = re.findall(r"(?:^|_)([012]\.\d{2})(?:_|$)", stem)
            if len(scale_matches) != 1:
                raise ValueError(f"Unrecognized scale in {name}: {scale_matches}")

            scale = float(scale_matches[0])
            symbols = [row[0] for row in atom_rows]
            positions = [[float(value) for value in row[1:4]] for row in atom_rows]
            records.setdefault(scale, {})[fragment] = {
                "name": name,
                "charge": int(parse_header_value(header, "charge")),
                "energy": float(parse_header_value(header, "dft_energy")),
                "symbols": symbols,
                "positions": positions,
            }
    return records


def read_benchmark_geometry(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        natoms = int(handle.readline())
        header = handle.readline().strip()
        atom_rows = [handle.readline().split() for _ in range(natoms)]
    return {
        "scale": float(parse_header_value(header, "scaling")),
        "benchmark": float(parse_header_value(header, "benchmark_Eint")),
        "charge": int(parse_header_value(header, "charge")),
        "charge_a": int(parse_header_value(header, "charge_a")),
        "charge_b": int(parse_header_value(header, "charge_b")),
        "symbols": [row[0] for row in atom_rows],
        "positions": [[float(value) for value in row[1:4]] for row in atom_rows],
    }


def assert_same_geometry(left: dict, right: dict, *, label: str) -> None:
    if left["symbols"] != right["symbols"]:
        raise ValueError(f"Symbol mismatch for {label}")
    for left_xyz, right_xyz in zip(left["positions"], right["positions"], strict=True):
        if max(abs(a - b) for a, b in zip(left_xyz, right_xyz, strict=True)) > 1e-8:
            raise ValueError(f"Coordinate mismatch for {label}")


def flatten_positions(positions: list[list[float]]) -> str:
    return " ".join(f"{value:.10f}" for xyz in positions for value in xyz)


def build_rows(source: Path) -> list[dict[str, str | int | float]]:
    gradient_root = source / "gradient" / "wB97M-D3BJ_def2-TZVPPD"
    geometry_root = source / "geometries"
    rows: list[dict[str, str | int | float]] = []

    for selection in SELECTIONS:
        system_id = selection["system_id"]
        curves = read_extended_xyz(gradient_root / selection["gradient_file"], system_id)
        if len(curves) != 10:
            raise ValueError(f"Expected 10 scales for {system_id}, found {len(curves)}")

        for scale in sorted(curves):
            fragments = curves[scale]
            if set(fragments) != {"ab", "a", "b"}:
                raise ValueError(f"Incomplete fragments for {system_id} at {scale}")

            scale_code = f"{round(scale * 100):03d}"
            geometry_name = f"{system_id}_{scale_code}.xyz"
            benchmark = read_benchmark_geometry(
                geometry_root / selection["geometry_dir"] / geometry_name
            )
            assert abs(benchmark["scale"] - scale) < 1e-12
            assert_same_geometry(fragments["ab"], benchmark, label=geometry_name)
            assert fragments["ab"]["charge"] == benchmark["charge"]
            assert fragments["a"]["charge"] == benchmark["charge_a"]
            assert fragments["b"]["charge"] == benchmark["charge_b"]

            natoms_a = len(fragments["a"]["symbols"])
            reconstructed_a = {
                "symbols": fragments["ab"]["symbols"][:natoms_a],
                "positions": fragments["ab"]["positions"][:natoms_a],
            }
            reconstructed_b = {
                "symbols": fragments["ab"]["symbols"][natoms_a:],
                "positions": fragments["ab"]["positions"][natoms_a:],
            }
            assert_same_geometry(fragments["a"], reconstructed_a, label=f"{system_id} A")
            assert_same_geometry(fragments["b"], reconstructed_b, label=f"{system_id} B")

            for fragment in ("ab", "a", "b"):
                record = fragments[fragment]
                rows.append(
                    {
                        "subset": selection["subset"],
                        "system_id": system_id,
                        "system_name": selection["system_name"],
                        "interaction_class": selection["interaction_class"],
                        "scale": f"{scale:.2f}",
                        "fragment": fragment.upper(),
                        "charge": record["charge"],
                        "natoms": len(record["symbols"]),
                        "symbols": " ".join(record["symbols"]),
                        "positions_angstrom": flatten_positions(record["positions"]),
                        "wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol": (
                            f"{record['energy']:.12f}"
                        ),
                        "ccsd_t_cbs_interaction_energy_kcal_mol": (
                            f"{benchmark['benchmark']:.12f}"
                        ),
                        "source_gradient_block": record["name"],
                        "source_geometry_file": geometry_name,
                    }
                )

    if len(rows) != 90:
        raise ValueError(f"Expected 90 rows, found {len(rows)}")
    return rows


def write_deterministic_csv_gz(rows: list[dict], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=FIELDNAMES, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="NCIAtlas Git checkout")
    parser.add_argument("output", type=Path, help="Output .csv.gz path")
    args = parser.parse_args()

    rows = build_rows(args.source.resolve())
    checksum = write_deterministic_csv_gz(rows, args.output.resolve())
    print(f"rows={len(rows)}")
    print(f"sha256={checksum}")
    print(f"source_revision={SOURCE_REVISION}")


if __name__ == "__main__":
    main()
