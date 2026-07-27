#!/usr/bin/env python3
"""Run the canonical NCI setup and diagnose one finite-difference force."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient


DEPENDENCY_CELL_IDS = (
    "setup",
    "tutorial-settings",
    "imports",
    "helper-imports",
    "load-nci-atlas",
    "configure-nci-model",
    "evaluate-nci-components",
    "compose-nci-pipeline",
)


def _indices_by_id(notebook: Any) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, cell in enumerate(notebook.cells):
        cell_id = str(cell.get("id", ""))
        if not cell_id or cell_id in indices:
            raise ValueError(f"missing or duplicate cell ID at index {index}: {cell_id!r}")
        indices[cell_id] = index
    missing = [cell_id for cell_id in DEPENDENCY_CELL_IDS if cell_id not in indices]
    if missing:
        raise ValueError(f"missing diagnostic dependency cells: {missing}")
    return indices


def _diagnostic_source(output_path: Path) -> str:
    return f'''
import json as _json
from pathlib import Path as _Path

_diagnostic_output = _Path({str(output_path)!r})
_diagnostic_output.parent.mkdir(parents=True, exist_ok=True)

_example_index = nci_graph_index.index[
    (nci_graph_index["system_id"] == "1.041")
    & np.isclose(nci_graph_index["scale"], 1.0)
    & (nci_graph_index["fragment"] == "AB")
].item()
_example = nci_atoms[_example_index]


def _full_result(atoms):
    output, _ = nci_pipeline_outputs([atoms])
    return (
        float(output["energy"].detach().reshape(-1)[0].cpu()),
        output["forces"].detach().cpu(),
        str(output["energy"].dtype),
    )


def _d3_result(atoms):
    batch = Batch.from_data_list([
        AtomicData.from_atoms(atoms, device="cpu", dtype=torch.float32)
    ], device=DEVICE)
    compute_neighbors(batch, config=nci_d3.model_config.neighbor_config)
    output = nci_d3(batch)
    return (
        float(output["energy"].detach().reshape(-1)[0].cpu()),
        output["forces"].detach().cpu(),
        str(output["energy"].dtype),
    )


_base_energy, _base_force, _full_energy_dtype = _full_result(_example)
_base_d3_energy, _base_d3_force, _d3_energy_dtype = _d3_result(_example)
_max_flat = int(_base_force.abs().reshape(-1).argmax())
_max_atom, _max_axis = divmod(_max_flat, 3)
_coordinates = ((0, 0, "original atom 0 x"), (_max_atom, _max_axis, "largest force component"))
_steps = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1)
_rows = []

for _atom_index, _axis, _label in _coordinates:
    _autograd_full = float(_base_force[_atom_index, _axis])
    _autograd_d3 = float(_base_d3_force[_atom_index, _axis])
    _autograd_core_coulomb = _autograd_full - _autograd_d3
    for _step in _steps:
        _energies = {{}}
        for _multiple in (-2.0, -1.0, 1.0, 2.0):
            _displaced = _example.copy()
            _displaced.positions[_atom_index, _axis] += _multiple * _step
            _full_energy, _, _ = _full_result(_displaced)
            _d3_energy, _, _ = _d3_result(_displaced)
            _energies[_multiple] = {{
                "full": _full_energy,
                "d3": _d3_energy,
                "core_coulomb": _full_energy - _d3_energy,
            }}

        _row = {{
            "coordinate": _label,
            "atom_index": _atom_index,
            "axis": _axis,
            "step_A": _step,
            "autograd_full_eV_A": _autograd_full,
            "autograd_d3_eV_A": _autograd_d3,
            "autograd_core_coulomb_eV_A": _autograd_core_coulomb,
            "energies_eV": {{str(key): value for key, value in _energies.items()}},
        }}
        for _component in ("full", "d3", "core_coulomb"):
            _force_2point = -(
                _energies[1.0][_component] - _energies[-1.0][_component]
            ) / (2.0 * _step)
            _force_5point = (
                _energies[2.0][_component]
                - 8.0 * _energies[1.0][_component]
                + 8.0 * _energies[-1.0][_component]
                - _energies[-2.0][_component]
            ) / (12.0 * _step)
            _row[f"fd2_{{_component}}_eV_A"] = _force_2point
            _row[f"fd5_{{_component}}_eV_A"] = _force_5point
        _rows.append(_row)

_repeat_energies = [_full_result(_example)[0] for _ in range(5)]
_report = {{
    "schema": "alchemi.part1-nci-force-diagnostic.v1",
    "gpu": torch.cuda.get_device_name(DEVICE),
    "torch_version": torch.__version__,
    "core_commit": installed_pins["Core"],
    "ops_commit": installed_pins["Ops"],
    "system_id": "1.041",
    "scale": 1.0,
    "fragment": "AB",
    "base_full_energy_eV": _base_energy,
    "base_d3_energy_eV": _base_d3_energy,
    "full_energy_dtype": _full_energy_dtype,
    "d3_energy_dtype": _d3_energy_dtype,
    "base_energy_repeats_eV": _repeat_energies,
    "base_force_eV_A": _base_force.tolist(),
    "base_d3_force_eV_A": _base_d3_force.tolist(),
    "coordinates": [
        {{"atom_index": atom, "axis": axis, "label": label}}
        for atom, axis, label in _coordinates
    ],
    "rows": _rows,
}}
_diagnostic_output.write_text(
    _json.dumps(_report, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
print("force diagnostic:", _diagnostic_output)
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executed-prefix", type=Path, required=True)
    parser.add_argument("--kernel", default="alchemi-main")
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    output_path = args.output.resolve()
    executed_path = args.executed_prefix.resolve()
    if len({notebook_path, output_path, executed_path}) != 3:
        raise ValueError("source notebook and diagnostic outputs must use distinct paths")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    executed_path.parent.mkdir(parents=True, exist_ok=True)

    notebook = nbformat.read(notebook_path, as_version=4)
    indices = _indices_by_id(notebook)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name=args.kernel,
        allow_errors=False,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.reset_execution_trackers()

    with client.setup_kernel():
        execution_count = 0
        for cell_id in DEPENDENCY_CELL_IDS:
            execution_count += 1
            index = indices[cell_id]
            client.execute_cell(
                notebook.cells[index], index, execution_count=execution_count
            )
            print(f"[{cell_id}] complete", flush=True)

        diagnostic = nbformat.v4.new_code_cell(_diagnostic_source(output_path))
        notebook.cells.append(diagnostic)
        try:
            client.execute_cell(
                diagnostic,
                len(notebook.cells) - 1,
                execution_count=execution_count + 1,
                store_history=False,
            )
        finally:
            notebook.cells.pop()

    nbformat.write(notebook, executed_path)
    if not output_path.is_file():
        raise RuntimeError("force diagnostic did not write its result")
    print(output_path.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
