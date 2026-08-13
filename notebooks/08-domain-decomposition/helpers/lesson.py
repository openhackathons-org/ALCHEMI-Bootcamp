"""Checked inputs, visuals, and evidence gates for Part 08."""

from __future__ import annotations

import html
import io
import json
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ase.build import bulk
from nvalchemi.data import AtomicData, Batch

TOOLKIT_COMMIT = "8c2c307c1c0c76baee6f7a68eb75a45da83ffd18"
TOOLKIT_OPS_COMMIT = "c1e23460859a784e1d78043bcd1c8af0d1095fa2"
BASE_STRUCTURE_SHA256 = (
    "5fcfc9394ebed3583267f20f322f60fb7b9311650e3b8dec4b8e8edaa4e0c0da"
)
BASE_MANIFEST_SHA256 = (
    "ea30e3f12f042f98f136147e783b56ab2e0da622f3486718b9fec69f3cde74b4"
)
CHECKPOINT_SHA256 = (
    "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
)
NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = NOTEBOOK_DIR.parents[1]
CAMPAIGN_TEMPLATE = NOTEBOOK_DIR / "external" / "campaign-spec.json"
CAMPAIGN_RESULT = NOTEBOOK_DIR / "results" / "current-pin" / "campaign.json"
EXPECTED_PINS = {
    "toolkit": {
        "distribution": "nvalchemi-toolkit",
        "version": "0.2.0",
        "commit": TOOLKIT_COMMIT,
    },
    "toolkit_ops": {
        "distribution": "nvalchemi-toolkit-ops",
        "version": "0.4.1",
        "commit": TOOLKIT_OPS_COMMIT,
    },
}


def repo_root(start: Path | None = None) -> Path:
    """Locate the tutorials checkout without relying on the working directory."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "environment" / "runtime-pins.toml").is_file()
            and (candidate / "shared" / "alchemi-dark.mplstyle").is_file()
        ):
            return candidate
    raise FileNotFoundError("Run this lesson from inside the tutorials v3 checkout.")


def configure_presentation() -> None:
    """Apply the shared plot style and compact dataframe defaults."""

    import matplotlib.pyplot as plt

    root = repo_root(Path(__file__))
    plt.style.use(root / "shared" / "alchemi-dark.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["svg.hashsalt"] = "part-08-domain-decomposition"
    pd.set_option("display.max_colwidth", 90)
    pd.set_option("display.width", 120)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    """Hash dtype, shape, and contiguous bytes of a NumPy array."""

    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _installed_commit(distribution_name: str) -> tuple[str, str]:
    package = distribution(distribution_name)
    direct_url_text = package.read_text("direct_url.json")
    if direct_url_text is None:
        return package.version, "unavailable"
    direct_url = json.loads(direct_url_text)
    return package.version, str(direct_url.get("vcs_info", {}).get("commit_id", ""))


def installed_pin_table() -> pd.DataFrame:
    """Read installed VCS provenance and require the lesson's frozen commits."""

    rows: list[dict[str, str]] = []
    for expected in EXPECTED_PINS.values():
        name = expected["distribution"]
        version, commit = _installed_commit(name)
        matches = version == expected["version"] and commit == expected["commit"]
        rows.append(
            {
                "distribution": name,
                "version": version,
                "installed VCS commit": commit,
                "pin check": "match" if matches else "mismatch",
            }
        )
    frame = pd.DataFrame(rows)
    if frame["pin check"].ne("match").any():
        raise RuntimeError("Installed Toolkit provenance does not match Part 08 pins.")
    return frame


def build_argon_control(device: torch.device | str = "cpu") -> Batch:
    """Build a small periodic 2×2×2 conventional-cell argon crystal."""

    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
    graph = AtomicData.from_atoms(atoms, device=device)
    graph.add_system_property(
        "energy",
        torch.zeros((1, 1), dtype=graph.positions.dtype, device=graph.positions.device),
    )
    graph.add_node_property("forces", torch.zeros_like(graph.positions))
    graph_fields = graph.model_dump(exclude_none=True)
    graph_fields["source_atom_id"] = torch.arange(
        len(atoms), dtype=torch.int64, device=graph.positions.device
    )
    graph = AtomicData(**graph_fields)
    return Batch.from_data_list(
        [graph],
        device=device,
        field_levels={"source_atom_id": "atom"},
    )


def base_box_identity(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Verify and summarize the checked Part 1 phenol/NMA base box."""

    directory = (
        root
        / "part-1-scalable-atomistic-workflows"
        / "data"
        / "domain_decomposition"
        / "prebuilt_base_box"
    )
    structure_path = directory / "structure.extxyz"
    manifest_path = directory / "manifest.json"
    structure_digest = sha256_file(structure_path)
    manifest_digest = sha256_file(manifest_path)
    if structure_digest != BASE_STRUCTURE_SHA256:
        raise RuntimeError(f"Unexpected base-box structure checksum: {structure_digest}")
    if manifest_digest != BASE_MANIFEST_SHA256:
        raise RuntimeError(f"Unexpected base-box manifest checksum: {manifest_digest}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    structure = manifest["structure"]
    if structure["sha256"] != structure_digest:
        raise RuntimeError("Base-box manifest and structure checksum disagree.")
    return {
        "system": "phenol + N-methylacetamide",
        "atom_count": int(structure["atom_count"]),
        "molecule_count": int(structure["molecule_count"]),
        "molecules_per_species": int(structure["molecules_per_species"]),
        "construction_density_g_cm3": float(structure["construction_density_g_cm3"]),
        "pbc": list(structure["pbc"]),
        "structure_sha256": structure_digest,
        "manifest_sha256": manifest_digest,
        "scope": manifest["interpretation"]["construction"],
    }


def control_summary(
    *,
    world_size: int,
    full_batch: Batch,
    owned_batch: Batch,
    gathered_batch: Batch | None,
) -> dict[str, Any]:
    """Summarize ownership without overstating the one-process fallback."""

    if world_size < 1:
        raise ValueError("world_size must be positive")
    if gathered_batch is None:
        raise ValueError("Call control_summary on the gather destination rank.")
    input_atoms = int(full_batch.num_nodes)
    owned_atoms = int(owned_batch.num_nodes)
    gathered_atoms = int(gathered_batch.num_nodes)
    if world_size == 1 and (owned_atoms != input_atoms or gathered_atoms != input_atoms):
        raise RuntimeError("One-process DomainParallel must preserve the full atom count.")
    return {
        "world size": world_size,
        "input atoms": input_atoms,
        "rank-0 owned atoms": owned_atoms,
        "gathered atoms": gathered_atoms,
        "spatially decomposed": world_size > 1,
        "interpretation": (
            "one-process control; no partition"
            if world_size == 1
            else "multi-rank spatial partition"
        ),
    }


def reorder_by_source_id(values: np.ndarray, source_ids: np.ndarray) -> np.ndarray:
    """Restore rank-contiguous gathered rows to original source-atom order."""

    array = np.asarray(values)
    ids = np.asarray(source_ids)
    if ids.ndim != 1 or array.shape[0] != ids.shape[0]:
        raise ValueError("source_atom_id must contain one ID per value row")
    expected = np.arange(ids.size, dtype=np.int64)
    if not np.array_equal(np.sort(ids.astype(np.int64, copy=False)), expected):
        raise ValueError("source_atom_id must be a permutation of 0..N-1")
    return array[np.argsort(ids, kind="stable")]


@dataclass(frozen=True)
class _AccessibleFigure:
    png: bytes
    svg: str
    alt: str

    def _repr_mimebundle_(self, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        safe_alt = html.escape(self.alt, quote=True)
        accessible_svg = self.svg.replace(
            ">",
            f'><title>{html.escape(self.alt)}</title>',
            1,
        )
        image = (
            f'<div role="img" aria-label="{safe_alt}" '
            f'style="max-width:100%;height:auto;">{accessible_svg}</div>'
        )
        data = {
            "text/html": image,
            "image/png": self.png,
            "text/plain": f"<AccessibleFigure alt={self.alt!r}>",
        }
        metadata = {"alt": self.alt, "image/png": {"alt": self.alt}}
        return data, metadata


def accessible_figure(figure: Any, alt: str) -> _AccessibleFigure:
    """Package a Matplotlib figure with renderer-visible alternative text."""

    if not alt.strip():
        raise ValueError("Figure alternative text must not be empty.")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    svg_buffer = io.StringIO()
    figure.savefig(
        svg_buffer,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    svg = svg_buffer.getvalue()
    svg = svg[svg.index("<svg") :]
    return _AccessibleFigure(png=buffer.getvalue(), svg=svg, alt=alt)


def plot_domain_ownership() -> Any:
    """Draw two spatial owners and the halo copies around their boundary."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    figure, axis = plt.subplots(figsize=(9.2, 4.5))
    axis.add_patch(Rectangle((0, 0), 5, 4, facecolor="#203542", edgecolor="#8AA5B5"))
    axis.add_patch(Rectangle((5, 0), 5, 4, facecolor="#28331D", edgecolor="#76B900"))
    axis.add_patch(
        Rectangle(
            (4, 0),
            2,
            4,
            facecolor="#D6A84A",
            edgecolor="#F2C66D",
            alpha=0.25,
            hatch="//",
        )
    )
    left = np.array([[0.8, 0.7], [1.8, 3.2], [3.2, 2.1], [4.6, 1.1], [4.8, 3.3]])
    right = np.array([[5.2, 0.8], [5.5, 2.8], [6.8, 1.9], [8.2, 3.2], [9.1, 0.7]])
    axis.scatter(left[:, 0], left[:, 1], s=65, color="#78B6D0", label="rank 0 owned")
    axis.scatter(right[:, 0], right[:, 1], s=65, color="#76B900", label="rank 1 owned")
    axis.scatter(
        right[:2, 0],
        right[:2, 1],
        s=110,
        facecolors="none",
        edgecolors="#78B6D0",
        linewidths=2,
        label="ghost copies on rank 0",
    )
    axis.scatter(
        left[-2:, 0],
        left[-2:, 1],
        s=110,
        facecolors="none",
        edgecolors="#76B900",
        linewidths=2,
        label="ghost copies on rank 1",
    )
    axis.axvline(5, color="#F3F4F6", linewidth=1.5)
    axis.text(2.5, 3.7, "rank 0 domain", ha="center", color="#D7EAF2")
    axis.text(7.5, 3.7, "rank 1 domain", ha="center", color="#DDEACF")
    axis.text(5.0, 0.2, "ghost width = cutoff + skin", ha="center", color="#F2C66D")
    axis.set_title("Owned atoms and ghost copies near one domain boundary")
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 4)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    figure.tight_layout()
    return figure


def plot_control_parity(reference: Batch, gathered: Batch) -> Any:
    """Compare direct and DomainParallel-fallback energy/force outputs."""

    import matplotlib.pyplot as plt

    direct_forces = reference.forces.detach().cpu().numpy().reshape(-1)
    gathered_forces = gathered.forces.detach().cpu().numpy().reshape(-1)
    residual = gathered_forces - direct_forces
    lower = float(min(direct_forces.min(), gathered_forces.min()))
    upper = float(max(direct_forces.max(), gathered_forces.max()))
    padding = max((upper - lower) * 0.08, 1.0e-9)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].scatter(direct_forces, gathered_forces, s=18, color="#76B900", alpha=0.8)
    axes[0].plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        color="#B9C0C8",
        linestyle="--",
    )
    axes[0].set_xlabel("Direct force component (eV/Å)")
    axes[0].set_ylabel("DomainParallel control (eV/Å)")
    axes[0].set_title("Force-component parity")
    if np.count_nonzero(residual) == 0:
        axes[1].axvline(0.0, color="#76B900", linewidth=3)
        axes[1].text(
            0.5,
            0.56,
            f"All {residual.size} components\nmatch exactly",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
        )
        axes[1].set_xlim(-1.0e-9, 1.0e-9)
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_yticks([])
    else:
        axes[1].hist(residual, bins=9, color="#76B900", edgecolor="#111315")
        axes[1].set_ylabel("Component count")
    axes[1].set_xlabel("Force residual (eV/Å)")
    energy_delta = float(
        (gathered.energy.detach() - reference.energy.detach()).abs().max().cpu()
    )
    axes[1].set_title(f"Maximum |ΔE| = {energy_delta:.2e} eV")
    figure.tight_layout()
    return figure


def campaign_spec_path() -> Path:
    """Prefer a locally recorded current-pin campaign over the template."""

    return CAMPAIGN_RESULT if CAMPAIGN_RESULT.is_file() else CAMPAIGN_TEMPLATE


def read_campaign_spec(path: Path | None = None) -> dict[str, Any]:
    """Read the campaign template or a completed local manifest."""

    selected = Path(path) if path is not None else campaign_spec_path()
    return json.loads(selected.read_text(encoding="utf-8"))


def campaign_workload_frame(campaign: dict[str, Any]) -> pd.DataFrame:
    """Shape the exact campaign workload into a compact display table."""

    workload = campaign["workload"]
    model = workload["model"]
    domain = workload["domain"]
    rows = [
        ("system", "phenol + N-methylacetamide"),
        ("atoms", f"{int(workload['atom_count']):,}"),
        ("molecules per species", f"{int(workload['molecules_per_species']):,}"),
        ("repeat factors", " × ".join(map(str, workload["repeat_factors_xyz"]))),
        ("periodic", str(bool(workload["periodic"]))),
        ("model adapter", model["adapter"]),
        ("model scope", model["scope"]),
        ("cutoff / skin", f"{domain['cutoff_a']} Å / {domain['skin_a']} Å"),
        ("required GPUs", ", ".join(map(str, campaign["required_world_sizes"]))),
    ]
    return pd.DataFrame(rows, columns=["field", "exact campaign value"])


@dataclass(frozen=True)
class CampaignReport:
    """Validated evidence state returned to the notebook."""

    ready: bool
    status: str
    reason: str
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    records: dict[int, dict[str, Any]] = field(default_factory=dict)


def _failed(
    status: str,
    reason: str,
    *,
    table: pd.DataFrame | None = None,
    records: dict[int, dict[str, Any]] | None = None,
) -> CampaignReport:
    return CampaignReport(
        ready=False,
        status=status,
        reason=reason,
        table=table if table is not None else pd.DataFrame(),
        records=records or {},
    )


def _safe_case_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("case path escapes the campaign directory")
    return candidate


def validate_campaign(
    campaign: dict[str, Any],
    *,
    root: Path | None = None,
) -> CampaignReport:
    """Validate completeness, provenance, ownership, artifacts, and parity."""

    status = str(campaign.get("status", "invalid"))
    if status != "complete":
        reason = str(
            campaign.get(
                "status_reason",
                "No complete current-pin 1/2/4-GPU campaign is available.",
            )
        )
        if "current-pin" not in reason:
            reason = f"{reason} No current-pin scaling evidence is reported."
        return _failed(status, reason)
    if campaign.get("schema") != "alchemi.part08-domain-campaign.v1":
        return _failed("invalid", "Campaign schema mismatch.")
    if campaign.get("current_pins") != EXPECTED_PINS:
        return _failed("invalid", "Campaign pin mismatch with the Part 08 lock.")

    required = [int(value) for value in campaign.get("required_world_sizes", [])]
    if required != [1, 2, 4]:
        return _failed("invalid", "Campaign must contain world sizes 1, 2, and 4.")
    cases = campaign.get("cases", {})
    if set(cases) != {"1", "2", "4"}:
        return _failed("incomplete", "Campaign is missing one or more required cases.")

    selected_root = Path(root) if root is not None else campaign_spec_path().parent
    workload = campaign["workload"]
    atom_count = int(workload["atom_count"])
    expected_passes = int(campaign["measurement"]["measured_pass_count"])
    artifacts: dict[int, np.ndarray] = {}
    records: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    input_tensor_sha: str | None = None

    for world_size in required:
        try:
            case_path = _safe_case_path(selected_root, str(cases[str(world_size)]))
            record = json.loads(case_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return _failed("invalid", f"Could not read {world_size}-GPU case: {error}")
        if (
            record.get("schema") != "alchemi.part08-domain-case.v1"
            or record.get("status") != "complete"
            or int(record.get("world_size", -1)) != world_size
        ):
            return _failed("invalid", f"Invalid {world_size}-GPU case header.")
        if record.get("current_pins") != campaign["current_pins"]:
            return _failed("invalid", f"Current-pin mismatch in {world_size}-GPU case.")

        input_record = record.get("input", {})
        if int(input_record.get("atom_count", -1)) != atom_count:
            return _failed("invalid", f"Atom-count mismatch in {world_size}-GPU case.")
        if input_record.get("base_structure_sha256") != workload["base_structure_sha256"]:
            return _failed("invalid", f"Input checksum mismatch in {world_size}-GPU case.")
        tensor_sha = str(input_record.get("tensor_sha256", ""))
        input_tensor_sha = input_tensor_sha or tensor_sha
        if not tensor_sha or tensor_sha != input_tensor_sha:
            return _failed("invalid", "The GPU cases did not use identical input tensors.")
        if tensor_sha != workload["input_tensor_sha256"]:
            return _failed(
                "invalid",
                f"Input tensor checksum mismatch in {world_size}-GPU case.",
            )

        model_record = record.get("model", {})
        if (
            model_record.get("alias") != workload["model"]["alias"]
            or model_record.get("checkpoint_sha256")
            != workload["model"]["checkpoint_sha256"]
        ):
            return _failed("invalid", f"Model identity mismatch in {world_size}-GPU case.")

        owned_counts = record.get("distributed", {}).get("owned_atom_counts", [])
        if (
            len(owned_counts) != world_size
            or any(int(count) < 0 for count in owned_counts)
            or sum(map(int, owned_counts)) != atom_count
        ):
            return _failed(
                "invalid", f"Incomplete rank ownership in {world_size}-GPU case."
            )

        timing = record.get("timing", {})
        pass_times = np.asarray(timing.get("pass_times_s", []), dtype=float)
        median_s = float(timing.get("median_s", np.nan))
        if (
            pass_times.size != expected_passes
            or not np.isfinite(pass_times).all()
            or np.any(pass_times <= 0)
            or not np.isclose(median_s, np.median(pass_times))
            or timing.get("ranks_synchronized") is not True
            or timing.get("publishable_benchmark") is not False
        ):
            return _failed("invalid", f"Timing contract failed for {world_size} GPUs.")

        gpu_names = record.get("runtime", {}).get("gpu_names", [])
        if len(gpu_names) != world_size or not all(gpu_names):
            return _failed("invalid", f"Runtime provenance missing for {world_size} GPUs.")
        rank_runtime = record.get("runtime", {}).get("ranks")
        if rank_runtime is not None and (
            len(rank_runtime) != world_size
            or [int(item["rank"]) for item in rank_runtime] != list(range(world_size))
            or [int(item["owned_atom_count"]) for item in rank_runtime]
            != list(map(int, owned_counts))
        ):
            return _failed(
                "invalid",
                f"Rank runtime ownership mismatch for {world_size} GPUs.",
            )

        output = record.get("output", {})
        try:
            artifact_path = _safe_case_path(case_path.parent, str(output["artifact"]))
            if sha256_file(artifact_path) != output["artifact_sha256"]:
                raise ValueError("artifact checksum mismatch")
            with np.load(artifact_path, allow_pickle=False) as saved:
                forces = np.asarray(saved["forces"])
                source_ids = np.asarray(saved["source_atom_id"])
        except (KeyError, OSError, ValueError) as error:
            return _failed(
                "invalid", f"Output artifact failed for {world_size} GPUs: {error}"
            )
        if (
            forces.shape != (atom_count, 3)
            or not np.isfinite(forces).all()
            or array_sha256(forces) != output.get("forces_sha256")
            or array_sha256(source_ids) != output.get("source_atom_id_sha256")
        ):
            return _failed("invalid", f"Force identity failed for {world_size} GPUs.")
        try:
            source_order_forces = reorder_by_source_id(forces, source_ids)
        except ValueError as error:
            return _failed("invalid", f"Source-atom identity failed: {error}")

        energy_ev = float(output.get("energy_ev", np.nan))
        if not np.isfinite(energy_ev):
            return _failed("invalid", f"Non-finite energy for {world_size} GPUs.")
        maximum_mic = float(output.get("maximum_mic_displacement_a", 0.0))
        if (
            not np.isfinite(maximum_mic)
            or maximum_mic
            > float(campaign["acceptance"]["position_mic_atol_a"])
        ):
            return _failed(
                "invalid",
                f"Position invariance failed for world size {world_size}.",
            )
        artifacts[world_size] = source_order_forces
        records[world_size] = record
        rows.append(
            {
                "GPUs": world_size,
                "owned atoms": sum(map(int, owned_counts)),
                "median time (s)": median_s,
                "energy (eV)": energy_ev,
            }
        )

    table = pd.DataFrame(rows).sort_values("GPUs").reset_index(drop=True)
    reference_energy = float(table.loc[table["GPUs"].eq(1), "energy (eV)"].iloc[0])
    reference_forces = artifacts[1]
    energy_atol = float(campaign["acceptance"]["energy_atol_ev_per_atom"])
    force_atol = float(campaign["acceptance"]["force_atol_ev_a"])
    force_rtol = float(campaign["acceptance"]["force_rtol"])
    energy_deltas: list[float] = []
    force_deltas: list[float] = []
    parity: list[str] = []

    for row in table.itertuples(index=False):
        world_size = int(row[0])
        energy_delta = abs(float(row[3]) - reference_energy) / atom_count
        force_delta = float(np.max(np.abs(artifacts[world_size] - reference_forces)))
        energy_deltas.append(energy_delta)
        force_deltas.append(force_delta)
        if world_size == 1:
            parity.append("reference")
            continue
        if energy_delta > energy_atol:
            table["ΔE / atom (eV)"] = energy_deltas + [np.nan] * (
                len(table) - len(energy_deltas)
            )
            return _failed(
                "invalid",
                f"energy parity failed for world size {world_size}.",
                table=table,
                records=records,
            )
        if not np.allclose(
            artifacts[world_size],
            reference_forces,
            rtol=force_rtol,
            atol=force_atol,
        ):
            return _failed(
                "invalid",
                f"force parity failed for world size {world_size}.",
                table=table,
                records=records,
            )
        parity.append("pass")

    reference_time = float(table.loc[table["GPUs"].eq(1), "median time (s)"].iloc[0])
    table["speedup"] = reference_time / table["median time (s)"]
    table["parallel efficiency"] = table["speedup"] / table["GPUs"]
    table["ΔE / atom (eV)"] = energy_deltas
    table["max |ΔF| (eV/Å)"] = force_deltas
    table["parity"] = parity
    return CampaignReport(
        ready=True,
        status="complete",
        reason=(
            "All required current-pin records, checksums, rank ownership, "
            "and energy/force parity checks passed."
        ),
        table=table,
        records=records,
    )


def evidence_status_frame(report: CampaignReport) -> pd.DataFrame:
    """Return a displayable status even when no campaign can be plotted."""

    return pd.DataFrame(
        [
            {
                "evidence status": report.status,
                "plot eligible": report.ready,
                "reason": report.reason,
            }
        ]
    )


def plot_campaign(report: CampaignReport) -> Any:
    """Plot timing and force parity only after the full evidence gate passes."""

    if not report.ready:
        raise RuntimeError("Campaign is not plot-eligible.")
    import matplotlib.pyplot as plt

    table = report.table
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(
        table["GPUs"],
        table["median time (s)"],
        marker="o",
        linewidth=2,
        color="#76B900",
    )
    axes[0].set_xticks(table["GPUs"])
    axes[0].set_xlabel("GPUs")
    axes[0].set_ylabel("Median evaluation time (s)")
    axes[0].set_title("Observed fixed-input time")
    axes[1].plot(
        table["GPUs"],
        table["max |ΔF| (eV/Å)"],
        marker="o",
        linewidth=2,
        color="#78B6D0",
    )
    axes[1].axhline(
        0.0,
        color="#B9C0C8",
        linestyle="--",
        linewidth=1,
    )
    axes[1].set_xticks(table["GPUs"])
    axes[1].set_xlabel("GPUs")
    axes[1].set_ylabel("Maximum force difference (eV/Å)")
    axes[1].set_title("Parity against one GPU")
    figure.tight_layout()
    return figure
