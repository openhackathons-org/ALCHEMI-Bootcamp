"""Portable repository and installed-package checks."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from typing import Any


def find_bootcamp_root(start: str | Path | None = None) -> Path:
    """Locate the Bootcamp root from any directory inside the checkout."""

    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "build" / "requirements.txt").is_file() and (
            candidate / "part-1-scalable-atomistic-workflows"
        ).is_dir():
            return candidate
    raise FileNotFoundError("Run this notebook inside the ALCHEMI-Bootcamp checkout.")


def installed_git_commit(distribution_name: str) -> str:
    """Read the immutable VCS revision recorded by Python packaging."""

    direct_url_text = metadata.distribution(distribution_name).read_text(
        "direct_url.json"
    )
    if direct_url_text is None:
        raise RuntimeError(f"{distribution_name} has no direct_url.json install record")
    direct_url = json.loads(direct_url_text)
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if not commit:
        raise RuntimeError(f"{distribution_name} is not installed from a Git pin")
    return str(commit)


def verify_toolkit_pins(core_commit: str, ops_commit: str) -> dict[str, str]:
    """Fail when installed Toolkit revisions differ from the tutorial pins."""

    installed = {
        "Core": installed_git_commit("nvalchemi-toolkit"),
        "Ops": installed_git_commit("nvalchemi-toolkit-ops"),
    }
    expected = {"Core": str(core_commit), "Ops": str(ops_commit)}
    if installed != expected:
        raise RuntimeError(
            "Installed Toolkit pins do not match this notebook: "
            f"Core={installed['Core']}, Ops={installed['Ops']}"
        )
    return installed


def check_batch_buffer_transfer(device: str | Any) -> dict[str, Any]:
    """Check the exact ``Batch.empty``/``Batch.put`` path used by stage buffers.

    The probe selects two differently sized graphs from a three-graph batch and
    checks float, integer, and boolean fields at both atom and system level. It
    also checks a second distinct ``put`` into the same buffer and a distinct
    ``put`` after ``zero()``. It returns a report instead of raising so the
    caller can record a useful error before allocating a large campaign
    workload.
    """

    import torch
    from nvalchemi.data import AtomicData, Batch

    target_device = torch.device(device)
    cases: list[dict[str, Any]] = []
    selected_graphs = (0, 2)
    graph_sizes = (2, 4, 3)

    for float_dtype in (torch.float32, torch.float64):
        graphs = []
        for graph_id, num_atoms in enumerate(graph_sizes):
            atom = torch.arange(num_atoms, dtype=float_dtype).unsqueeze(1)
            positions = torch.cat(
                (atom, atom + 0.25, atom + 0.5), dim=1
            ) + graph_id * 10.0
            graph = AtomicData(
                positions=positions,
                atomic_numbers=torch.tensor(
                    [1 + ((graph_id + index) % 8) for index in range(num_atoms)],
                    dtype=torch.int64,
                ),
                energy=torch.tensor([[graph_id + 0.5]], dtype=float_dtype),
                pbc=torch.tensor(
                    [[graph_id % 2 == 0, graph_id % 3 == 0, False]],
                    dtype=torch.bool,
                ),
            )
            graph.add_node_property(
                "atom_code",
                torch.arange(num_atoms, dtype=torch.int32) + graph_id * 100,
            )
            graph.add_system_property(
                "system_code",
                torch.tensor([[graph_id + 1_000]], dtype=torch.int32),
            )
            graph.add_system_property(
                "system_code64",
                torch.tensor([[(1 << 40) + graph_id]], dtype=torch.int64),
            )
            graphs.append(graph)

        source = Batch.from_data_list(graphs, device=target_device)
        source_ptr = source.batch_ptr.detach().cpu().tolist()

        def new_destination() -> Any:
            return Batch.empty(
                num_systems=3,
                num_nodes=sum(graph_sizes[index] for index in selected_graphs),
                num_edges=0,
                template=source,
                device=target_device,
            )

        def payload_mismatches(
            destination: Any,
            expected_graphs: tuple[int, ...],
        ) -> list[str]:
            observed: list[str] = []
            if destination.num_graphs != len(expected_graphs):
                observed.append("num_graphs")
            expected_sizes = [graph_sizes[index] for index in expected_graphs]
            if destination.num_nodes_list != expected_sizes:
                observed.append("num_nodes_list")

            atom_slices = [
                slice(source_ptr[index], source_ptr[index + 1])
                for index in expected_graphs
            ]
            atom_count = sum(expected_sizes)
            node_fields = ("positions", "atomic_numbers", "atom_code")
            for field in node_fields:
                expected = torch.cat(
                    [source[field][atom_slice] for atom_slice in atom_slices], dim=0
                )
                actual = destination[field][:atom_count]
                if actual.dtype != expected.dtype or not torch.equal(actual, expected):
                    observed.append(field)

            system_fields = ("energy", "pbc", "system_code", "system_code64")
            graph_index = torch.tensor(
                expected_graphs, dtype=torch.long, device=target_device
            )
            for field in system_fields:
                expected = source[field][graph_index]
                actual = destination[field][: len(expected_graphs)]
                if actual.dtype != expected.dtype or not torch.equal(actual, expected):
                    observed.append(field)
            return observed

        checks: dict[str, dict[str, Any]] = {}

        source_mask = torch.tensor(
            [True, False, True], dtype=torch.bool, device=target_device
        )
        copied_mask = torch.zeros(3, dtype=torch.bool, device=target_device)
        destination_mask = torch.zeros(3, dtype=torch.bool, device=target_device)
        destination = new_destination()
        first_put_mismatches: list[str] = []
        try:
            destination.put(
                source,
                source_mask,
                copied_mask=copied_mask,
                dest_mask=destination_mask,
            )
        except Exception as exc:
            first_put_mismatches.append(f"raised {type(exc).__name__}: {exc}")
        else:
            expected_destination_mask = torch.tensor(
                [True, True, False], dtype=torch.bool, device=target_device
            )
            if not torch.equal(copied_mask, source_mask):
                first_put_mismatches.append("copied_mask")
            if not torch.equal(destination_mask, expected_destination_mask):
                first_put_mismatches.append("destination_mask")
            first_put_mismatches.extend(
                payload_mismatches(
                    destination,
                    selected_graphs,
                )
            )
        checks["first_put"] = {
            "passed": not first_put_mismatches,
            "mismatches": sorted(set(first_put_mismatches)),
        }

        destination = new_destination()
        destination_mask.zero_()
        repeated_put_mismatches: list[str] = []
        for step, graph_index in enumerate(selected_graphs, start=1):
            step_mask = torch.zeros(3, dtype=torch.bool, device=target_device)
            step_mask[graph_index] = True
            step_copied_mask = torch.zeros_like(step_mask)
            try:
                destination.put(
                    source,
                    step_mask,
                    copied_mask=step_copied_mask,
                    dest_mask=destination_mask,
                )
            except Exception as exc:
                repeated_put_mismatches.append(
                    f"put_{step}_raised {type(exc).__name__}: {exc}"
                )
                break
            if not torch.equal(step_copied_mask, step_mask):
                repeated_put_mismatches.append(f"put_{step}_copied_mask")
        expected_repeated_mask = torch.tensor(
            [True, True, False], dtype=torch.bool, device=target_device
        )
        if not torch.equal(destination_mask, expected_repeated_mask):
            repeated_put_mismatches.append("destination_mask")
        repeated_put_mismatches.extend(
            payload_mismatches(
                destination,
                selected_graphs,
            )
        )
        checks["repeated_put"] = {
            "passed": not repeated_put_mismatches,
            "mismatches": sorted(set(repeated_put_mismatches)),
        }

        destination = new_destination()
        destination_mask.zero_()
        first_mask = torch.tensor(
            [True, False, False], dtype=torch.bool, device=target_device
        )
        first_copied_mask = torch.zeros_like(first_mask)
        zero_then_put_mismatches: list[str] = []
        try:
            destination.put(
                source,
                first_mask,
                copied_mask=first_copied_mask,
                dest_mask=destination_mask,
            )
            destination.zero()
            destination_mask.zero_()
            second_mask = torch.tensor(
                [False, False, True], dtype=torch.bool, device=target_device
            )
            second_copied_mask = torch.zeros_like(second_mask)
            destination.put(
                source,
                second_mask,
                copied_mask=second_copied_mask,
                dest_mask=destination_mask,
            )
        except Exception as exc:
            zero_then_put_mismatches.append(
                f"raised {type(exc).__name__}: {exc}"
            )
        else:
            if not torch.equal(first_copied_mask, first_mask):
                zero_then_put_mismatches.append("first_copied_mask")
            if not torch.equal(second_copied_mask, second_mask):
                zero_then_put_mismatches.append("second_copied_mask")
            expected_after_zero_mask = torch.tensor(
                [True, False, False], dtype=torch.bool, device=target_device
            )
            if not torch.equal(destination_mask, expected_after_zero_mask):
                zero_then_put_mismatches.append("destination_mask")
            zero_then_put_mismatches.extend(
                payload_mismatches(
                    destination,
                    (selected_graphs[1],),
                )
            )
        checks["zero_then_put"] = {
            "passed": not zero_then_put_mismatches,
            "mismatches": sorted(set(zero_then_put_mismatches)),
        }

        mismatches = sorted(
            f"{check_name}.{mismatch}"
            for check_name, check in checks.items()
            for mismatch in check["mismatches"]
        )

        cases.append(
            {
                "float_dtype": str(float_dtype).removeprefix("torch."),
                "passed": not mismatches,
                "mismatches": mismatches,
                "checks": checks,
            }
        )

    return {
        "device": str(target_device),
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


__all__ = [
    "check_batch_buffer_transfer",
    "find_bootcamp_root",
    "installed_git_commit",
    "verify_toolkit_pins",
]
