"""Build and repeat the periodic phenol/N-methylacetamide box used in Part 1.

Packmol creates an initial non-overlapping arrangement.  It does not
equilibrate the box and the requested density is therefore called the
``construction density`` throughout this module.

The recorded domain-decomposition campaign runs Packmol only for one checked
128-pair base box. Larger power-of-two inputs are deterministic integer
supercells of that box. This keeps the molecular arrangement, composition, and
construction density fixed while avoiding repeated large Packmol jobs.

The helpers accept ASE structures or the tutorial's NCI Atlas table and do not
import NVIDIA ALCHEMI Toolkit.  This keeps the notebook focused on the Toolkit
model and domain-decomposition APIs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
from typing import Any

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.neighborlist import neighbor_list
import numpy as np
import pandas as pd

from ..nci_atlas import row_to_atoms
from .config import DOMAIN_METHODOLOGY


NCI_SYSTEM_ID = DOMAIN_METHODOLOGY.nci_system_id
NCI_SCALE = DOMAIN_METHODOLOGY.nci_scale
COMPONENT_NAMES = ("phenol", "N-methylacetamide")
ATOMIC_MASS_VOLUME_FACTOR = 1.66053906660
BASE_PAIR_COUNT = DOMAIN_METHODOLOGY.live_molecules_per_species
BASE_ATOM_COUNT = (
    BASE_PAIR_COUNT * DOMAIN_METHODOLOGY.atoms_per_composition_unit
)
REQUIRED_SUPERCELL_ARRAYS = (
    "source_atom_id",
    "molecule_id",
    "molecule_component",
    "molecule_kind",
    "template_atom_index",
)


@dataclass(frozen=True)
class MoleculeTemplate:
    """One named neutral or charged molecular template."""

    name: str
    atoms: Atoms
    charge_e: int

    @property
    def atom_count(self) -> int:
        return len(self.atoms)

    @property
    def mass_u(self) -> float:
        return float(np.sum(self.atoms.get_masses()))


@dataclass(frozen=True)
class MolecularBoxPlan:
    """All input-derived values and configured Packmol settings for one box."""

    templates: tuple[MoleculeTemplate, MoleculeTemplate]
    molecule_counts: tuple[int, int]
    construction_density_g_cm3: float
    box_length_a: float
    packmol_tolerance_a: float
    packmol_precision_a: float
    packmol_seed: int
    system_id: str = NCI_SYSTEM_ID
    scale: float = NCI_SCALE

    @property
    def molecules_per_species(self) -> int:
        if self.molecule_counts[0] != self.molecule_counts[1]:
            raise ValueError("the phenol/N-methylacetamide plan is not 1:1")
        return self.molecule_counts[0]

    @property
    def molecule_count(self) -> int:
        return int(sum(self.molecule_counts))

    @property
    def atom_count(self) -> int:
        return int(
            sum(
                count * template.atom_count
                for template, count in zip(
                    self.templates, self.molecule_counts, strict=True
                )
            )
        )

    @property
    def total_mass_u(self) -> float:
        return float(
            sum(
                count * template.mass_u
                for template, count in zip(
                    self.templates, self.molecule_counts, strict=True
                )
            )
        )

    @property
    def net_charge_e(self) -> int:
        return int(
            sum(
                count * template.charge_e
                for template, count in zip(
                    self.templates, self.molecule_counts, strict=True
                )
            )
        )


@dataclass(frozen=True)
class MolecularBoxValidation:
    """Checks performed on a Packmol output before it reaches Toolkit."""

    molecule_count: int
    atom_count: int
    construction_density_g_cm3: float
    density_from_mass_and_cell_g_cm3: float
    net_charge_e: int
    packmol_precision_a: float
    periodic_min_distance_a: float | None
    periodic_min_distance_lower_bound_a: float
    min_distance_required_a: float


@dataclass(frozen=True)
class PackmolProcessResult:
    """Small runner result accepted by :func:`build_nci_molecular_box`."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


PackmolRunner = Callable[
    [str | Path | None, Path, str],
    PackmolProcessResult | subprocess.CompletedProcess[str],
]


def _positive_finite(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    try:
        unchanged = float(value) == number
    except (TypeError, ValueError):
        unchanged = False
    if number <= 0 or not unchanged:
        raise ValueError(f"{name} must be a positive integer")
    return number


def balanced_repeat_factors(
    *,
    base_pair_count: int,
    target_pair_count: int,
) -> tuple[int, int, int]:
    """Return balanced x/y/z repeats for a power-of-two size increase.

    Powers of two are distributed cyclically from z to x. The resulting
    factors are ordered and differ by no more than a factor of two:
    ``(1, 1, 2)``, ``(1, 2, 2)``, ``(2, 2, 2)``, and so on.
    """

    base_count = _positive_integer(base_pair_count, name="base_pair_count")
    target_count = _positive_integer(target_pair_count, name="target_pair_count")
    if target_count < base_count or target_count % base_count:
        raise ValueError(
            "target_pair_count must be an integer multiple of base_pair_count"
        )
    multiplier = target_count // base_count
    if multiplier & (multiplier - 1):
        raise ValueError(
            "target_pair_count/base_pair_count must be a power of two"
        )
    exponent = multiplier.bit_length() - 1
    quotient, remainder = divmod(exponent, 3)
    factors = [2**quotient] * 3
    for axis in range(3 - remainder, 3):
        factors[axis] *= 2
    return tuple(factors)


def build_molecular_supercell(
    base_atoms: Atoms,
    *,
    base_pair_count: int,
    target_pair_count: int,
) -> tuple[Atoms, tuple[int, int, int]]:
    """Repeat one checked periodic molecular box and preserve stable IDs.

    ASE repeats every per-atom array with the coordinates. ``source_atom_id``
    and ``molecule_id`` additionally receive an image-dependent offset so they
    remain exact, unique integer identities in the expanded structure. All
    other arrays, including component and template labels, retain their base
    values.
    """

    if not isinstance(base_atoms, Atoms):
        raise TypeError("base_atoms must be an ASE Atoms object")
    if not len(base_atoms):
        raise ValueError("base_atoms must not be empty")
    if not np.asarray(base_atoms.pbc, dtype=bool).all():
        raise ValueError("base_atoms must be periodic in x, y, and z")
    cell = np.asarray(base_atoms.cell, dtype=float)
    if (
        cell.shape != (3, 3)
        or not np.isfinite(cell).all()
        or not math.isfinite(float(base_atoms.get_volume()))
        or float(base_atoms.get_volume()) <= 0.0
    ):
        raise ValueError("base_atoms must have a finite, nonzero 3D cell")

    missing = [
        name for name in REQUIRED_SUPERCELL_ARRAYS if name not in base_atoms.arrays
    ]
    if missing:
        raise ValueError(
            "base_atoms is missing required arrays: " + ", ".join(missing)
        )

    base_count = _positive_integer(base_pair_count, name="base_pair_count")
    if len(base_atoms) % base_count:
        raise ValueError("base atom count is not divisible by base_pair_count")
    source_ids = np.asarray(base_atoms.arrays["source_atom_id"], dtype=np.int64)
    expected_source_ids = np.arange(len(base_atoms), dtype=np.int64)
    if not np.array_equal(source_ids, expected_source_ids):
        raise ValueError("base source_atom_id must be ordered from 0 to N-1")

    molecule_ids = np.asarray(base_atoms.arrays["molecule_id"], dtype=np.int64)
    if molecule_ids.shape != (len(base_atoms),):
        raise ValueError("base molecule_id must contain one value per atom")
    unique_molecule_ids = np.unique(molecule_ids)
    expected_molecule_ids = np.arange(2 * base_count, dtype=np.int64)
    if not np.array_equal(unique_molecule_ids, expected_molecule_ids):
        raise ValueError(
            "base molecule_id must cover the two species with contiguous IDs"
        )

    repeat_factors = balanced_repeat_factors(
        base_pair_count=base_count,
        target_pair_count=target_pair_count,
    )
    repeat_count = math.prod(repeat_factors)
    supercell = base_atoms.repeat(repeat_factors)

    source_offsets = np.repeat(
        np.arange(repeat_count, dtype=np.int64) * len(base_atoms),
        len(base_atoms),
    )
    molecule_offsets = np.repeat(
        np.arange(repeat_count, dtype=np.int64) * len(unique_molecule_ids),
        len(base_atoms),
    )
    supercell.set_array(
        "source_atom_id",
        np.tile(source_ids, repeat_count) + source_offsets,
    )
    supercell.set_array(
        "molecule_id",
        np.tile(molecule_ids, repeat_count) + molecule_offsets,
    )

    expected_atoms = len(base_atoms) * repeat_count
    if len(supercell) != expected_atoms:
        raise RuntimeError("ASE returned an unexpected supercell atom count")
    if not np.array_equal(
        supercell.arrays["source_atom_id"],
        np.arange(expected_atoms, dtype=np.int64),
    ):
        raise RuntimeError("expanded source_atom_id is not ordered from 0 to N-1")
    if len(np.unique(supercell.arrays["molecule_id"])) != 2 * target_pair_count:
        raise RuntimeError("expanded molecule_id values are not unique per molecule")
    if not np.asarray(supercell.pbc, dtype=bool).all():
        raise RuntimeError("ASE repeat did not preserve three-dimensional PBC")

    base_density = (
        float(np.sum(base_atoms.get_masses()))
        * ATOMIC_MASS_VOLUME_FACTOR
        / float(base_atoms.get_volume())
    )
    supercell_density = (
        float(np.sum(supercell.get_masses()))
        * ATOMIC_MASS_VOLUME_FACTOR
        / float(supercell.get_volume())
    )
    if not math.isclose(
        supercell_density,
        base_density,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("integer repeats changed the construction density")

    supercell.info.update(
        {
            "base_pair_count": base_count,
            "pair_count": target_pair_count,
            "molecules_per_species": target_pair_count,
            "supercell_repeat_x": repeat_factors[0],
            "supercell_repeat_y": repeat_factors[1],
            "supercell_repeat_z": repeat_factors[2],
        }
    )
    return supercell, repeat_factors


def _integer_charge(value: Any, *, label: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} charge must be an integer") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label} charge must be an integer")
    return int(number)


def _template_from_atoms(
    name: str, atoms: Atoms, charge_e: int | None
) -> MoleculeTemplate:
    if not isinstance(atoms, Atoms) or len(atoms) == 0:
        raise ValueError(f"{name} must be a nonempty ASE Atoms template")
    template = atoms.copy()
    template.set_pbc(False)
    template.set_cell([0.0, 0.0, 0.0])
    if not np.isfinite(template.positions).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    if charge_e is None:
        if "charge" not in template.info:
            raise ValueError(f"{name} needs an integer total charge")
        charge_e = template.info["charge"]
    charge = _integer_charge(charge_e, label=name)
    template.info["charge"] = charge
    return MoleculeTemplate(name=name, atoms=template, charge_e=charge)


def _templates_from_nci_table(
    table: pd.DataFrame,
    *,
    system_id: str,
    scale: float,
) -> tuple[MoleculeTemplate, ...]:
    if not isinstance(table, pd.DataFrame):
        raise TypeError("NCI Atlas input must be a pandas DataFrame")
    required = {
        "system_id",
        "scale",
        "fragment",
        "charge",
        "natoms",
        "symbols",
        "positions_angstrom",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"NCI Atlas table is missing {sorted(missing)!r}")

    scales = pd.to_numeric(table["scale"], errors="coerce").to_numpy(dtype=float)
    selected = table.loc[
        table["system_id"].astype(str).eq(system_id)
        & np.isclose(scales, scale, rtol=0.0, atol=1.0e-12)
    ]
    if len(selected) != 3 or set(selected["fragment"].astype(str)) != {"AB", "A", "B"}:
        raise ValueError(
            f"NCI Atlas system {system_id} at scale {scale:g} "
            "must contain AB, A, and B"
        )
    records = {str(row["fragment"]): row for row in selected.to_dict(orient="records")}
    structures = {
        fragment: row_to_atoms(record) for fragment, record in records.items()
    }
    combined = structures["A"] + structures["B"]
    if combined.get_chemical_symbols() != structures[
        "AB"
    ].get_chemical_symbols() or not np.array_equal(
        combined.positions, structures["AB"].positions
    ):
        raise ValueError("NCI Atlas fragments A and B do not reconstruct AB")
    charge_a = _integer_charge(records["A"]["charge"], label="fragment A")
    charge_b = _integer_charge(records["B"]["charge"], label="fragment B")
    charge_ab = _integer_charge(records["AB"]["charge"], label="fragment AB")
    if charge_a + charge_b != charge_ab:
        raise ValueError("NCI Atlas fragment charges do not sum to AB")

    return (
        _template_from_atoms(COMPONENT_NAMES[0], structures["A"], charge_a),
        _template_from_atoms(COMPONENT_NAMES[1], structures["B"], charge_b),
    )


def _normalize_templates(
    templates: pd.DataFrame | Sequence[Atoms | MoleculeTemplate],
    *,
    system_id: str,
    scale: float,
) -> tuple[MoleculeTemplate, MoleculeTemplate]:
    if isinstance(templates, pd.DataFrame):
        resolved = _templates_from_nci_table(
            templates,
            system_id=system_id,
            scale=scale,
        )
    else:
        if len(templates) != 2:
            raise ValueError("provide phenol and N-methylacetamide templates")
        resolved_items: list[MoleculeTemplate] = []
        for name, item in zip(COMPONENT_NAMES, templates, strict=True):
            if isinstance(item, MoleculeTemplate):
                if item.name != name:
                    raise ValueError(
                        f"expected template {name!r}, received {item.name!r}"
                    )
                resolved_items.append(
                    _template_from_atoms(item.name, item.atoms, item.charge_e)
                )
            else:
                resolved_items.append(_template_from_atoms(name, item, None))
        resolved = tuple(resolved_items)
    if len(resolved) != 2:
        raise ValueError("expected exactly two molecular templates")
    return resolved[0], resolved[1]


def plan_nci_molecular_box(
    templates: pd.DataFrame | Sequence[Atoms | MoleculeTemplate],
    *,
    nci_system_id: str = NCI_SYSTEM_ID,
    nci_scale: float = NCI_SCALE,
    molecules_per_species: int,
    construction_density_g_cm3: float,
    packmol_tolerance_a: float,
    packmol_precision_a: float,
    packmol_seed: int,
) -> MolecularBoxPlan:
    """Plan a cubic 1:1 phenol/N-methylacetamide construction.

    ``templates`` may be the complete tutorial NCI Atlas table or two ASE
    templates ordered as phenol then N-methylacetamide.  The cubic box length is
    derived from their ASE atomic masses, the requested molecule count, and the
    configured construction density. Packmol distance tolerance and optimizer
    precision are both explicit so the generated input and validation use the
    same acceptance limit.
    """

    if not isinstance(nci_system_id, str) or not nci_system_id.strip():
        raise ValueError("nci_system_id must be a nonempty string")
    system_id = nci_system_id.strip()
    scale = _positive_finite(nci_scale, name="nci_scale")
    count = _positive_integer(
        molecules_per_species,
        name="molecules_per_species",
    )
    density = _positive_finite(
        construction_density_g_cm3,
        name="construction_density_g_cm3",
    )
    tolerance = _positive_finite(packmol_tolerance_a, name="packmol_tolerance_a")
    precision = _positive_finite(packmol_precision_a, name="packmol_precision_a")
    if precision >= tolerance:
        raise ValueError("packmol_precision_a must be smaller than packmol_tolerance_a")
    if isinstance(packmol_seed, bool):
        raise ValueError("packmol_seed must be an integer")
    try:
        seed = int(packmol_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("packmol_seed must be an integer") from exc
    if seed != packmol_seed:
        raise ValueError("packmol_seed must be an integer")

    resolved = _normalize_templates(
        templates,
        system_id=system_id,
        scale=scale,
    )
    if any(template.charge_e != 0 for template in resolved):
        raise ValueError(
            "the phenol/N-methylacetamide lesson requires two neutral templates"
        )
    total_mass_u = count * sum(template.mass_u for template in resolved)
    volume_a3 = total_mass_u * ATOMIC_MASS_VOLUME_FACTOR / density
    box_length_a = volume_a3 ** (1.0 / 3.0)
    if not math.isfinite(box_length_a):
        raise ValueError("derived box length is not finite")

    return MolecularBoxPlan(
        templates=resolved,
        molecule_counts=(count, count),
        construction_density_g_cm3=density,
        box_length_a=box_length_a,
        packmol_tolerance_a=tolerance,
        packmol_precision_a=precision,
        packmol_seed=seed,
        system_id=system_id,
        scale=scale,
    )


def render_packmol_input(
    plan: MolecularBoxPlan,
    *,
    template_filenames: Mapping[str, str | Path],
    output_filename: str | Path,
) -> str:
    """Render deterministic Packmol input with one global cubic PBC command."""

    missing = set(COMPONENT_NAMES) - set(template_filenames)
    if missing:
        raise ValueError(f"template filenames are missing {sorted(missing)!r}")
    output = Path(output_filename)
    if output.is_absolute() or output.name != str(output):
        raise ValueError("Packmol output filename must be a simple relative name")
    lines = [
        f"tolerance {plan.packmol_tolerance_a:.8f}",
        f"precision {plan.packmol_precision_a:.8f}",
        "filetype xyz",
        f"output {output.as_posix()}",
        f"seed {plan.packmol_seed}",
        (
            f"pbc {plan.box_length_a:.8f} "
            f"{plan.box_length_a:.8f} {plan.box_length_a:.8f}"
        ),
        "",
    ]
    for template, count in zip(plan.templates, plan.molecule_counts, strict=True):
        filename = Path(template_filenames[template.name])
        if filename.is_absolute() or filename.name != str(filename):
            raise ValueError("Packmol template filenames must be simple relative names")
        lines.extend(
            (
                f"structure {filename.as_posix()}",
                f"  number {count}",
                "end structure",
                "",
            )
        )
    return "\n".join(lines)


def _subprocess_packmol_runner(
    packmol_binary: str | Path | None,
    cwd: Path,
    input_text: str,
) -> PackmolProcessResult:
    if packmol_binary is None:
        raise ValueError("packmol_binary is required for subprocess execution")
    input_path = cwd / "packmol.inp"
    if not input_path.is_file() or input_path.read_text(encoding="utf-8") != input_text:
        raise RuntimeError("saved Packmol input does not match the requested run")
    # Packmol rewinds standard input during setup, so it needs a seekable file
    # rather than a subprocess pipe.
    with input_path.open("r", encoding="utf-8") as input_stream:
        completed = subprocess.run(
            [str(packmol_binary)],
            cwd=cwd,
            stdin=input_stream,
            capture_output=True,
            check=False,
            text=True,
        )
    return PackmolProcessResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _expected_output_layout(
    plan: MolecularBoxPlan,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    symbols: list[str] = []
    molecule_ids: list[int] = []
    component_ids: list[int] = []
    molecule_id = 0
    for component_id, (template, count) in enumerate(
        zip(plan.templates, plan.molecule_counts, strict=True)
    ):
        template_symbols = template.atoms.get_chemical_symbols()
        for _ in range(count):
            symbols.extend(template_symbols)
            molecule_ids.extend([molecule_id] * len(template_symbols))
            component_ids.extend([component_id] * len(template_symbols))
            molecule_id += 1
    return (
        symbols,
        np.asarray(molecule_ids, dtype=np.int64),
        np.asarray(component_ids, dtype=np.int8),
    )


def _periodic_inter_molecular_minimum(
    atoms: Atoms,
    *,
    search_cutoff_a: float,
) -> float | None:
    molecule_ids = np.asarray(atoms.arrays["molecule_id"], dtype=np.int64)
    first, second, distances = neighbor_list(
        "ijd",
        atoms,
        cutoff=search_cutoff_a,
        self_interaction=False,
    )
    between_molecules = molecule_ids[first] != molecule_ids[second]
    if not np.any(between_molecules):
        return None
    return float(np.min(np.asarray(distances)[between_molecules]))


def validate_molecular_box(
    plan: MolecularBoxPlan,
    atoms: Atoms,
) -> MolecularBoxValidation:
    """Validate counts, PBC, density, charge, and intermolecular separation.

    Packmol considers a target distance satisfied within its configured
    precision, so the minimum accepted separation is ``tolerance - precision``.
    """

    if not isinstance(atoms, Atoms):
        raise TypeError("Packmol output must be an ASE Atoms object")
    expected_symbols, molecule_ids, component_ids = _expected_output_layout(plan)
    if len(atoms) != plan.atom_count:
        raise ValueError(
            f"Packmol returned {len(atoms)} atoms; expected {plan.atom_count}"
        )
    if atoms.get_chemical_symbols() != expected_symbols:
        raise ValueError("Packmol output atom order or molecular composition changed")
    if not np.asarray(atoms.pbc, dtype=bool).all():
        raise ValueError("Packmol output must be periodic in x, y, and z")
    cell = np.asarray(atoms.cell, dtype=float)
    expected_cell = np.eye(3) * plan.box_length_a
    if cell.shape != (3, 3) or not np.allclose(
        cell, expected_cell, rtol=0.0, atol=1.0e-7
    ):
        raise ValueError("Packmol output cell does not match the planned cubic box")

    reported_charge = atoms.info.get("charge", plan.net_charge_e)
    if _integer_charge(reported_charge, label="packed box") != plan.net_charge_e:
        raise ValueError(
            "Packmol output net charge does not match the construction plan"
        )
    atoms.arrays["molecule_id"] = molecule_ids
    atoms.arrays["molecule_component"] = component_ids
    atoms.info["charge"] = plan.net_charge_e
    atoms.info["construction_density_g_cm3"] = plan.construction_density_g_cm3
    atoms.info["packmol_precision_a"] = plan.packmol_precision_a
    atoms.info["nci_system_id"] = plan.system_id
    atoms.info["nci_scale"] = plan.scale

    density_from_mass_and_cell = (
        float(np.sum(atoms.get_masses()))
        * ATOMIC_MASS_VOLUME_FACTOR
        / float(atoms.get_volume())
    )
    if not math.isclose(
        density_from_mass_and_cell,
        plan.construction_density_g_cm3,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Packmol output density does not match the construction plan")
    search_cutoff = plan.packmol_tolerance_a + 0.25
    observed_minimum = _periodic_inter_molecular_minimum(
        atoms,
        search_cutoff_a=search_cutoff,
    )
    lower_bound = search_cutoff if observed_minimum is None else observed_minimum
    required = plan.packmol_tolerance_a - plan.packmol_precision_a
    if lower_bound < required:
        raise ValueError(
            "Packmol output contains an intermolecular contact shorter than "
            f"{required:.6g} Å"
        )

    return MolecularBoxValidation(
        molecule_count=plan.molecule_count,
        atom_count=plan.atom_count,
        construction_density_g_cm3=plan.construction_density_g_cm3,
        density_from_mass_and_cell_g_cm3=density_from_mass_and_cell,
        net_charge_e=plan.net_charge_e,
        packmol_precision_a=plan.packmol_precision_a,
        periodic_min_distance_a=observed_minimum,
        periodic_min_distance_lower_bound_a=lower_bound,
        min_distance_required_a=required,
    )


def build_nci_molecular_box(
    plan: MolecularBoxPlan,
    workdir: str | Path,
    *,
    packmol_binary: str | Path | None = None,
    runner: PackmolRunner | None = None,
    extxyz_path: str | Path | None = None,
) -> Atoms:
    """Run Packmol through an explicit binary or injected runner.

    The working directory retains the two molecular templates, rendered Packmol
    input, raw XYZ output, and optional validated extxyz file for inspection.
    A non-zero Packmol exit is an error even if a best-effort XYZ file exists.
    """

    if runner is None and packmol_binary is None:
        raise ValueError("provide packmol_binary or an explicit Packmol runner")
    execute = _subprocess_packmol_runner if runner is None else runner
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    template_filenames = {
        COMPONENT_NAMES[0]: "phenol.xyz",
        COMPONENT_NAMES[1]: "n_methylacetamide.xyz",
    }
    for template in plan.templates:
        ase_write(
            root / template_filenames[template.name],
            template.atoms,
            format="xyz",
        )
    output_filename = "phenol_n_methylacetamide_box.xyz"
    input_text = render_packmol_input(
        plan,
        template_filenames=template_filenames,
        output_filename=output_filename,
    )
    (root / "packmol.inp").write_text(input_text, encoding="utf-8")
    process = execute(packmol_binary, root, input_text)
    returncode = int(process.returncode)
    stdout = str(process.stdout or "")
    stderr = str(process.stderr or "")
    (root / "packmol.stdout.log").write_text(stdout, encoding="utf-8")
    (root / "packmol.stderr.log").write_text(stderr, encoding="utf-8")
    output_path = root / output_filename
    if returncode != 0:
        raise RuntimeError(
            f"Packmol exited with status {returncode}: "
            f"{(stderr or stdout).strip()[-1000:]}"
        )
    if not output_path.is_file():
        raise RuntimeError("Packmol completed without writing the requested XYZ file")

    atoms = ase_read(output_path, format="xyz")
    atoms.set_cell([plan.box_length_a] * 3)
    atoms.set_pbc(True)
    validation = validate_molecular_box(plan, atoms)
    atoms.info.update(
        {
            "packmol_seed": plan.packmol_seed,
            "packmol_tolerance_a": plan.packmol_tolerance_a,
            "packmol_precision_a": plan.packmol_precision_a,
            "periodic_min_distance_lower_bound_a": (
                validation.periodic_min_distance_lower_bound_a
            ),
        }
    )
    if validation.periodic_min_distance_a is not None:
        atoms.info["periodic_min_distance_a"] = validation.periodic_min_distance_a
    if extxyz_path is not None:
        destination = Path(extxyz_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ase_write(destination, atoms, format="extxyz")
    return atoms


def box_summary_table(
    plan: MolecularBoxPlan,
    atoms: Atoms | None = None,
) -> pd.DataFrame:
    """Return a concise learner-facing table of planned and checked values."""

    validation = validate_molecular_box(plan, atoms) if atoms is not None else None
    rows: list[dict[str, Any]] = []
    for template, count in zip(plan.templates, plan.molecule_counts, strict=True):
        rows.append(
            {
                "component": template.name,
                "molecules": count,
                "atoms_per_molecule": template.atom_count,
                "charge_per_molecule_e": template.charge_e,
            }
        )
    rows.append(
        {
            "component": "periodic box",
            "molecules": plan.molecule_count,
            "atoms_per_molecule": pd.NA,
            "charge_per_molecule_e": pd.NA,
            "atoms_total": plan.atom_count,
            "net_charge_e": plan.net_charge_e,
            "box_length_a": plan.box_length_a,
            "construction_density_g_cm3": plan.construction_density_g_cm3,
            "packmol_tolerance_a": plan.packmol_tolerance_a,
            "packmol_precision_a": plan.packmol_precision_a,
            "density_from_mass_and_cell_g_cm3": (
                validation.density_from_mass_and_cell_g_cm3 if validation else pd.NA
            ),
            "min_distance_required_a": (
                validation.min_distance_required_a
                if validation
                else plan.packmol_tolerance_a - plan.packmol_precision_a
            ),
            "periodic_min_distance_a_or_lower_bound": (
                validation.periodic_min_distance_lower_bound_a if validation else pd.NA
            ),
        }
    )
    return pd.DataFrame(rows)


def molecule_charge_tables(
    plan: MolecularBoxPlan,
    atoms: Atoms,
    molecule_charges: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label and summarize predicted charge sums for every packed molecule.

    ``molecule_charges`` is the output of a Toolkit segmented sum over
    ``atoms.arrays["molecule_id"]``. The model constrains the total periodic
    graph charge, not the charge of each source molecule, so these tables are a
    diagnostic rather than a per-molecule neutrality check.
    """

    if not isinstance(plan, MolecularBoxPlan):
        raise TypeError("plan must be a MolecularBoxPlan")
    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an ASE Atoms object")
    missing = {
        name
        for name in ("molecule_id", "molecule_component")
        if name not in atoms.arrays
    }
    if missing:
        raise ValueError(
            "packed atoms are missing required arrays: " + ", ".join(sorted(missing))
        )

    values = molecule_charges
    for method_name in ("detach", "cpu"):
        method = getattr(values, method_name, None)
        if callable(method):
            values = method()
    to_numpy = getattr(values, "numpy", None)
    if callable(to_numpy):
        values = to_numpy()
    charges = np.asarray(values, dtype=float)
    if charges.ndim == 2 and charges.shape[1] == 1:
        charges = charges[:, 0]
    if charges.shape != (plan.molecule_count,):
        raise ValueError(
            "molecule_charges must contain one value per packed molecule"
        )
    if not np.isfinite(charges).all():
        raise ValueError("molecule_charges contains a non-finite value")

    molecule_ids = np.asarray(atoms.arrays["molecule_id"], dtype=np.int64)
    component_ids = np.asarray(
        atoms.arrays["molecule_component"],
        dtype=np.int64,
    )
    if molecule_ids.shape != (len(atoms),) or component_ids.shape != (len(atoms),):
        raise ValueError("molecule label arrays must contain one value per atom")
    expected_ids = np.arange(plan.molecule_count, dtype=np.int64)
    if not np.array_equal(np.unique(molecule_ids), expected_ids):
        raise ValueError("molecule_id must cover every packed molecule exactly")

    labels = pd.DataFrame(
        {
            "molecule_id": molecule_ids,
            "component_id": component_ids,
        }
    )
    component_counts = labels.groupby("molecule_id")["component_id"].nunique()
    if not bool((component_counts == 1).all()):
        raise ValueError("one molecule_id maps to multiple molecular components")
    component_by_molecule = (
        labels.groupby("molecule_id")["component_id"]
        .first()
        .reindex(expected_ids)
        .to_numpy(dtype=np.int64)
    )
    if np.any((component_by_molecule < 0) | (component_by_molecule >= len(plan.templates))):
        raise ValueError("molecule_component is outside the construction plan")

    table = pd.DataFrame(
        {
            "molecule_id": expected_ids,
            "component": [
                plan.templates[index].name for index in component_by_molecule
            ],
            "predicted_charge_e": charges,
        }
    )

    def summarize(component: str, group: pd.Series) -> dict[str, Any]:
        values = group.to_numpy(dtype=float)
        return {
            "component": component,
            "molecules": len(values),
            "mean_charge_e": float(np.mean(values)),
            "mean_abs_charge_e": float(np.mean(np.abs(values))),
            "standard_deviation_e": float(np.std(values, ddof=0)),
            "minimum_charge_e": float(np.min(values)),
            "maximum_charge_e": float(np.max(values)),
            "total_charge_e": float(np.sum(values)),
        }

    summary_rows = [
        summarize(name, group["predicted_charge_e"])
        for name, group in table.groupby("component", sort=False)
    ]
    summary_rows.append(summarize("all molecules", table["predicted_charge_e"]))
    return table, pd.DataFrame(summary_rows)


__all__ = (
    "BASE_ATOM_COUNT",
    "BASE_PAIR_COUNT",
    "MolecularBoxPlan",
    "MolecularBoxValidation",
    "MoleculeTemplate",
    "PackmolProcessResult",
    "PackmolRunner",
    "balanced_repeat_factors",
    "box_summary_table",
    "build_molecular_supercell",
    "build_nci_molecular_box",
    "molecule_charge_tables",
    "plan_nci_molecular_box",
    "render_packmol_input",
    "validate_molecular_box",
)
