"""Load and reduce the small NCI Atlas subset used in Part 1.

This module works with ASE structures, NumPy arrays, and pandas tables only.
Toolkit ``AtomicData`` and ``Batch`` construction stays in the notebook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

from ase import Atoms
import numpy as np
import pandas as pd


NCI_ATLAS_SUBSET_SHA256 = (
    "7ffbc071e2998cee8e487a2697517187110a05f436920f8611d28d2af5d4d7b7"
)
EXPECTED_SCALES = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.25, 1.50, 2.00)
EXPECTED_SYSTEMS = {
    "1.041": (
        "HB375x10",
        "phenol - N-methylacetamide",
        "neutral hydrogen bond",
    ),
    "1.07.74": (
        "D442x10",
        "propyne - methyl azide",
        "dispersion-dominated",
    ),
    "08.007": (
        "IHB100x10",
        "ammonia - benzoate",
        "ionic hydrogen bond",
    ),
}
FRAGMENTS = ("AB", "A", "B")
CURVE_KEY_COLUMNS = (
    "subset",
    "system_id",
    "system_name",
    "interaction_class",
    "scale",
)
GRAPH_INDEX_COLUMNS = (
    *CURVE_KEY_COLUMNS,
    "fragment",
    "charge",
    "natoms",
    "source_gradient_block",
    "source_geometry_file",
)
REQUIRED_COLUMNS = (
    *GRAPH_INDEX_COLUMNS,
    "symbols",
    "positions_angstrom",
    "wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol",
    "ccsd_t_cbs_interaction_energy_kcal_mol",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _parse_structure_fields(
    record: Mapping[str, Any], *, label: str
) -> tuple[list[str], np.ndarray, int]:
    try:
        natoms_value = float(record["natoms"])
        symbols = str(record["symbols"]).split()
        positions = np.fromstring(str(record["positions_angstrom"]), sep=" ")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} has invalid structure fields") from exc
    if not np.isfinite(natoms_value) or not natoms_value.is_integer():
        raise ValueError(f"{label} has a non-integer atom count")
    natoms = int(natoms_value)
    if natoms <= 0 or len(symbols) != natoms:
        raise ValueError(f"{label} atom count does not match its symbols")
    if positions.size != 3 * natoms or not np.isfinite(positions).all():
        raise ValueError(f"{label} atom count does not match its coordinates")
    return symbols, positions.reshape(natoms, 3), natoms


def validate_nci_atlas_subset(table: pd.DataFrame) -> None:
    """Check the exact 90-row, three-system tutorial subset."""

    if not isinstance(table, pd.DataFrame):
        raise TypeError("NCI Atlas data must be a pandas DataFrame")
    missing = set(REQUIRED_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"NCI Atlas table is missing {sorted(missing)!r}")
    if len(table) != 90:
        raise ValueError(f"NCI Atlas table has {len(table)} rows; expected 90")

    text_columns = (
        "subset",
        "system_id",
        "system_name",
        "interaction_class",
        "fragment",
        "symbols",
        "positions_angstrom",
        "source_gradient_block",
        "source_geometry_file",
    )
    for column in text_columns:
        values = table[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"{column} contains a missing or empty value")

    numeric_columns = (
        "scale",
        "charge",
        "natoms",
        "wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol",
        "ccsd_t_cbs_interaction_energy_kcal_mol",
    )
    numeric: dict[str, np.ndarray] = {}
    for column in numeric_columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains a non-finite value")
        numeric[column] = values
    for column in ("charge", "natoms"):
        if not np.equal(numeric[column], np.rint(numeric[column])).all():
            raise ValueError(f"{column} must contain integers")
    if (numeric["natoms"] <= 0).any() or (numeric["scale"] <= 0.0).any():
        raise ValueError("atom counts and separation scales must be positive")

    duplicate_keys = table.duplicated(["system_id", "scale", "fragment"])
    if duplicate_keys.any():
        raise ValueError("system, scale, and fragment keys must be unique")
    if table["source_gradient_block"].nunique() != 90:
        raise ValueError("source gradient-block identifiers must be unique")
    if table["source_geometry_file"].nunique() != 30:
        raise ValueError("expected one source geometry file for each of 30 curves")

    system_rows = table[
        ["system_id", "subset", "system_name", "interaction_class"]
    ].drop_duplicates()
    if system_rows["system_id"].duplicated().any():
        raise ValueError("system metadata changes between rows")
    observed_systems = {
        str(row.system_id): (row.subset, row.system_name, row.interaction_class)
        for row in system_rows.itertuples(index=False)
    }
    if observed_systems != EXPECTED_SYSTEMS:
        raise ValueError("NCI Atlas system identities do not match the tutorial subset")

    for system_id, group in table.groupby("system_id", sort=False):
        scales = np.sort(pd.to_numeric(group["scale"]).unique())
        if scales.shape != (10,) or not np.allclose(
            scales, EXPECTED_SCALES, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"{system_id} does not contain the expected ten scales")

    grouped = table.groupby(list(CURVE_KEY_COLUMNS), sort=False, dropna=False)
    if grouped.ngroups != 30:
        raise ValueError(f"NCI Atlas table has {grouped.ngroups} curves; expected 30")
    for key, group in grouped:
        label = f"{key[1]} at scale {key[-1]:g}"
        if set(group["fragment"]) != set(FRAGMENTS) or len(group) != 3:
            raise ValueError(f"{label} must contain one AB, A, and B row")
        if group["source_geometry_file"].nunique() != 1:
            raise ValueError(f"{label} fragments do not share a source geometry")
        if group["ccsd_t_cbs_interaction_energy_kcal_mol"].nunique() != 1:
            raise ValueError(f"{label} fragments do not share the CCSD(T)/CBS value")

        records = {str(row.fragment): row._asdict() for row in group.itertuples()}
        structures = {
            fragment: _parse_structure_fields(record, label=f"{label} {fragment}")
            for fragment, record in records.items()
        }
        ab_symbols, ab_positions, ab_count = structures["AB"]
        a_symbols, a_positions, a_count = structures["A"]
        b_symbols, b_positions, b_count = structures["B"]
        if ab_count != a_count + b_count or ab_symbols != a_symbols + b_symbols:
            raise ValueError(f"{label} monomer symbols do not reconstruct the dimer")
        if not np.array_equal(ab_positions, np.vstack((a_positions, b_positions))):
            raise ValueError(f"{label} monomer coordinates do not reconstruct the dimer")
        charges = {
            fragment: int(round(float(record["charge"])))
            for fragment, record in records.items()
        }
        if charges["AB"] != charges["A"] + charges["B"]:
            raise ValueError(f"{label} fragment charges do not sum to the dimer charge")


def load_nci_atlas_subset(
    path: str | Path,
    *,
    expected_sha256: str | None = NCI_ATLAS_SUBSET_SHA256,
) -> pd.DataFrame:
    """Load the packaged subset and check its checksum and scientific layout."""

    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"NCI Atlas subset not found: {data_path}")
    if expected_sha256 is not None:
        observed = _sha256(data_path)
        if observed != expected_sha256:
            raise ValueError(
                f"NCI Atlas subset SHA-256 mismatch: {observed}; "
                f"expected {expected_sha256}"
            )
    table = pd.read_csv(
        data_path,
        dtype={
            "subset": "string",
            "system_id": "string",
            "system_name": "string",
            "interaction_class": "string",
            "fragment": "string",
            "symbols": "string",
            "positions_angstrom": "string",
            "source_gradient_block": "string",
            "source_geometry_file": "string",
        },
    )
    validate_nci_atlas_subset(table)
    return table


def row_to_atoms(record: Mapping[str, Any]) -> Atoms:
    """Convert one subset row to a nonperiodic ASE structure."""

    label = str(record.get("source_gradient_block", "NCI Atlas row"))
    symbols, positions, _ = _parse_structure_fields(record, label=label)
    try:
        charge_value = float(record["charge"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} has an invalid total charge") from exc
    if not np.isfinite(charge_value) or not charge_value.is_integer():
        raise ValueError(f"{label} has a non-integer total charge")

    atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
    for field in (
        "subset",
        "system_id",
        "system_name",
        "interaction_class",
        "scale",
        "fragment",
        "source_gradient_block",
        "source_geometry_file",
    ):
        if field in record:
            atoms.info[field] = _native_scalar(record[field])
    atoms.info["charge"] = int(charge_value)
    return atoms


def rows_to_atoms(table: pd.DataFrame) -> list[Atoms]:
    """Convert the validated rows to ASE structures in unchanged row order."""

    validate_nci_atlas_subset(table)
    return [row_to_atoms(row) for row in table.to_dict(orient="records")]


def build_graph_index(table: pd.DataFrame) -> pd.DataFrame:
    """Return row-aligned graph metadata for notebook-built Toolkit objects."""

    validate_nci_atlas_subset(table)
    index = table[list(GRAPH_INDEX_COLUMNS)].copy().reset_index(drop=True)
    index.insert(0, "graph_index", np.arange(len(index), dtype=np.int64))
    return index


def _ordered_graph_index(graph_index: pd.DataFrame) -> pd.DataFrame:
    required = {"graph_index", *CURVE_KEY_COLUMNS, "fragment"}
    missing = required - set(graph_index.columns)
    if missing:
        raise ValueError(f"graph index is missing {sorted(missing)!r}")
    indices = pd.to_numeric(graph_index["graph_index"], errors="coerce").to_numpy(
        dtype=float
    )
    expected = np.arange(len(graph_index), dtype=float)
    if (
        not np.isfinite(indices).all()
        or not np.equal(indices, np.rint(indices)).all()
        or not np.array_equal(np.sort(indices), expected)
    ):
        raise ValueError("graph_index must be a permutation of 0 through N-1")
    ordered = graph_index.assign(graph_index=indices.astype(np.int64)).sort_values(
        "graph_index"
    )
    grouped = ordered.groupby(list(CURVE_KEY_COLUMNS), sort=False, dropna=False)
    for key, group in grouped:
        if len(group) != 3 or set(group["fragment"]) != set(FRAGMENTS):
            raise ValueError(f"curve {key!r} must contain one AB, A, and B graph")
    return ordered.reset_index(drop=True)


def _reduce_one_member(
    graph_index: pd.DataFrame,
    components: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    curve_order = graph_index[list(CURVE_KEY_COLUMNS)].drop_duplicates(
        ignore_index=True
    )
    curve_index = pd.MultiIndex.from_frame(curve_order)
    result = curve_order.copy()
    for name, values in components.items():
        frame = graph_index[list(CURVE_KEY_COLUMNS) + ["fragment"]].copy()
        frame[name] = values
        wide = frame.pivot(
            index=list(CURVE_KEY_COLUMNS), columns="fragment", values=name
        )
        interaction = (wide["AB"] - wide["A"] - wide["B"]).reindex(curve_index)
        result[name] = interaction.to_numpy(dtype=float)
    return result


def reduce_fragment_energies(
    graph_index: pd.DataFrame,
    components: Mapping[str, Any],
    *,
    unit_scale: float = 1.0,
) -> pd.DataFrame:
    """Reduce AB/A/B graph energies to interaction curves.

    Every component must have shape ``(graphs,)`` or ``(members, graphs)``.
    Components are already composed before this call, so choices such as adding
    D3 or Coulomb remain explicit in the notebook.
    """

    ordered = _ordered_graph_index(graph_index)
    if not components:
        raise ValueError("at least one graph-energy component is required")
    scale = float(unit_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("unit_scale must be finite and positive")

    arrays: dict[str, np.ndarray] = {}
    expected_shape: tuple[int, ...] | None = None
    for name, values in components.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("component names must be non-empty strings")
        array = np.asarray(values, dtype=float)
        if array.ndim not in (1, 2) or array.shape[-1] != len(ordered):
            raise ValueError(
                f"{name!r} must have shape (graphs,) or (members, graphs)"
            )
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError("all graph-energy components must have the same shape")
        if not np.isfinite(array).all():
            raise ValueError(f"{name!r} contains a non-finite graph energy")
        arrays[name] = array * scale

    assert expected_shape is not None
    if len(expected_shape) == 1:
        return _reduce_one_member(ordered, arrays)

    member_curves = []
    for member in range(expected_shape[0]):
        member_values = {name: values[member] for name, values in arrays.items()}
        curves = _reduce_one_member(ordered, member_values)
        curves.insert(len(CURVE_KEY_COLUMNS), "member", member)
        member_curves.append(curves)
    return pd.concat(member_curves, ignore_index=True)


def extract_repeated_interaction_reference(
    table: pd.DataFrame,
    source_column: str,
    *,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Extract a reference interaction value repeated on AB, A, and B rows."""

    required = {*CURVE_KEY_COLUMNS, "fragment", source_column}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"reference table is missing {sorted(missing)!r}")
    values = pd.to_numeric(table[source_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source_column} contains a non-finite value")
    grouped = table.groupby(list(CURVE_KEY_COLUMNS), sort=False, dropna=False)
    for key, group in grouped:
        if len(group) != 3 or set(group["fragment"]) != set(FRAGMENTS):
            raise ValueError(f"curve {key!r} must contain one AB, A, and B row")
        if group[source_column].nunique() != 1:
            raise ValueError(f"{source_column} differs between fragments for {key!r}")
    name = source_column if output_column is None else output_column
    return (
        grouped[source_column]
        .first()
        .rename(name)
        .reset_index()
    )


def mean_member_curves(
    member_curves: pd.DataFrame,
    component_columns: Sequence[str],
    *,
    spread_component: str,
    spread_column: str | None = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """Average ensemble curves and report the member spread for one component."""

    components = tuple(component_columns)
    if not components or len(set(components)) != len(components):
        raise ValueError("component_columns must contain unique names")
    required = {*CURVE_KEY_COLUMNS, "member", *components, spread_component}
    missing = required - set(member_curves.columns)
    if missing:
        raise ValueError(f"member curves are missing {sorted(missing)!r}")
    if not isinstance(ddof, int) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    if member_curves.duplicated([*CURVE_KEY_COLUMNS, "member"]).any():
        raise ValueError("member curves contain duplicate curve/member rows")

    members_by_curve = member_curves.groupby(
        list(CURVE_KEY_COLUMNS), sort=False, dropna=False
    )["member"].agg(lambda values: tuple(sorted(values)))
    if members_by_curve.empty or members_by_curve.nunique() != 1:
        raise ValueError("every curve must contain the same ensemble members")
    member_count = len(members_by_curve.iloc[0])
    if member_count <= ddof:
        raise ValueError("not enough ensemble members for the requested ddof")

    numeric = member_curves[list(set(components) | {spread_component})].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("member curves contain a non-finite value")
    prepared = member_curves.copy()
    prepared[numeric.columns] = numeric
    grouped = prepared.groupby(list(CURVE_KEY_COLUMNS), sort=False, dropna=False)
    result = grouped[list(components)].mean().reset_index()
    output_name = spread_column or f"{spread_component}_std"
    if output_name in result.columns:
        raise ValueError(f"spread column {output_name!r} collides with a mean column")
    result[output_name] = grouped[spread_component].std(ddof=ddof).to_numpy()
    return result


def interaction_metrics(
    curves: pd.DataFrame,
    comparisons: Mapping[str, tuple[str, str]],
    *,
    group_column: str = "system_name",
    mean_columns: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Calculate per-system mean absolute errors and optional column means."""

    if not comparisons:
        raise ValueError("at least one metric comparison is required")
    requested = {group_column}
    for prediction, reference in comparisons.values():
        requested.update((prediction, reference))
    requested.update((mean_columns or {}).values())
    missing = requested - set(curves.columns)
    if missing:
        raise ValueError(f"curve table is missing {sorted(missing)!r}")

    numeric_columns = requested - {group_column}
    numeric = curves[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("metric inputs contain a non-finite value")
    prepared = curves.copy()
    prepared[numeric.columns] = numeric

    rows: list[dict[str, Any]] = []
    for group_name, group in prepared.groupby(group_column, sort=False):
        row: dict[str, Any] = {group_column: group_name}
        for label, (prediction, reference) in comparisons.items():
            row[label] = float(np.mean(np.abs(group[prediction] - group[reference])))
        for label, column in (mean_columns or {}).items():
            if label in row:
                raise ValueError(f"metric label {label!r} is duplicated")
            row[label] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index(group_column)


def assemble_nci_comparison_curves(
    graph_index: pd.DataFrame,
    reference_data: pd.DataFrame,
    component_energies_eV: Mapping[str, Any],
    *,
    d3_graph_energies_eV: Any,
    dft_total_energy_column: str,
    cc_interaction_energy_column: str,
    comparisons: Mapping[str, tuple[str, str]],
    energy_to_kcal_mol: float,
    spread_component: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply AB-A-B, merge both references, and calculate named errors.

    The caller supplies the component names and comparison pairs so the
    scientific choices remain visible in the notebook.  This helper owns only
    the repeated reductions and one-to-one table joins.
    """

    if spread_component not in component_energies_eV:
        raise ValueError("spread_component must name one supplied model component")
    if dft_total_energy_column not in reference_data:
        raise ValueError(f"missing DFT reference column {dft_total_energy_column!r}")
    if cc_interaction_energy_column not in reference_data:
        raise ValueError(f"missing CC reference column {cc_interaction_energy_column!r}")

    member_curves = reduce_fragment_energies(
        graph_index,
        component_energies_eV,
        unit_scale=float(energy_to_kcal_mol),
    )
    curves = mean_member_curves(
        member_curves,
        tuple(component_energies_eV),
        spread_component=spread_component,
    )

    dft = reduce_fragment_energies(
        graph_index,
        {"dft_full": reference_data[dft_total_energy_column]},
    )
    d3_interaction = reduce_fragment_energies(
        graph_index,
        {"d3_interaction": d3_graph_energies_eV},
        unit_scale=float(energy_to_kcal_mol),
    )
    dft["dft_no_d3"] = dft["dft_full"] - d3_interaction["d3_interaction"]
    cc = extract_repeated_interaction_reference(
        reference_data,
        cc_interaction_energy_column,
        output_column="ccsd_t_cbs",
    )
    for reference in (dft, cc):
        curves = curves.merge(
            reference,
            on=list(CURVE_KEY_COLUMNS),
            validate="one_to_one",
        )

    metrics = interaction_metrics(
        curves,
        comparisons,
        mean_columns={"ensemble spread": f"{spread_component}_std"},
    )
    return member_curves, curves, metrics


__all__ = [
    "CURVE_KEY_COLUMNS",
    "EXPECTED_SCALES",
    "EXPECTED_SYSTEMS",
    "FRAGMENTS",
    "GRAPH_INDEX_COLUMNS",
    "NCI_ATLAS_SUBSET_SHA256",
    "assemble_nci_comparison_curves",
    "build_graph_index",
    "extract_repeated_interaction_reference",
    "interaction_metrics",
    "load_nci_atlas_subset",
    "mean_member_curves",
    "reduce_fragment_energies",
    "row_to_atoms",
    "rows_to_atoms",
    "validate_nci_atlas_subset",
]
