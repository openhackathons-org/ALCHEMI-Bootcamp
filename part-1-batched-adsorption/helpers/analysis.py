"""Analysis helpers for the AdsorbML MACE-vs-DFT tutorial.

This module keeps the validation contract separate from notebook glue:

* adsorption energies are always in eV per adsorbate,
* negative values mean exothermic binding,
* final relaxed sites are classified geometrically instead of trusting
  the starting configuration label,
* reference values marked as ``context`` are kept out of strict parity
  statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config_search import Configuration, fcc111_site_candidates, sites_for_host
from .models import OptimizationResult, atomic_data_to_ase
from .references import AdsorbMLReference, LITERATURE_OC157_MAD_GUIDE_EV

if TYPE_CHECKING:
    import ase


ADSORPTION_ENERGY_FORMULA = (
    "E_ads = E_slab+ads - E_clean_slab - E_gas_ads; "
    "negative values are exothermic binding energies in eV per adsorbate"
)

CANONICAL_EADS_COLUMN = "E_ads (eV)"
LEGACY_EBIND_COLUMN = "E_bind (eV)"


def adsorption_energy_column(df: pd.DataFrame) -> str:
    """Return the canonical adsorption-energy column, with legacy fallback."""
    if CANONICAL_EADS_COLUMN in df.columns:
        return CANONICAL_EADS_COLUMN
    return LEGACY_EBIND_COLUMN


@dataclass(frozen=True)
class FinalSiteAnalysis:
    """Geometric classification of one relaxed adsorbate-slab structure."""

    label: str
    host: str
    adsorbate: str
    start_site: str
    start_orientation: str
    final_site: str
    final_site_distance_A: float
    binding_atom_symbol: str
    binding_atom_index: int
    binding_height_A: float
    nearest_surface_distance_A: float
    tilt_deg: float | None


@dataclass(frozen=True)
class ValidationResult:
    """MACE-vs-reference result for the best relaxed configuration of a pair."""

    host: str
    adsorbate: str
    pair: str
    tier: str
    reference_scope: str
    MACE_site: str
    reference_site: str
    site_match: bool | None
    E_MACE_eV: float
    E_ref_eV: float | None
    delta_E_eV: float | None
    abs_delta_over_MAD: float | None
    status: str
    notes: str


def compute_adsorption_energy_ev(
    e_slab_ads_ev: float,
    e_clean_slab_ev: float,
    e_gas_ads_ev: float,
) -> float:
    """Return E_ads in eV per adsorbate; negative means exothermic."""
    return float(e_slab_ads_ev) - float(e_clean_slab_ev) - float(e_gas_ads_ev)


def _xy_delta_minimum_image(
    p_xy: np.ndarray,
    q_xy: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """Minimum-image xy displacement between two points in a slab cell."""
    delta = np.asarray(p_xy, dtype=float) - np.asarray(q_xy, dtype=float)
    basis = np.vstack([cell[0, :2], cell[1, :2]]).T
    try:
        frac = np.linalg.solve(basis, delta)
    except np.linalg.LinAlgError:
        return delta
    frac -= np.round(frac)
    return basis @ frac


def _nearest_site_name(
    anchor_position: np.ndarray,
    site_map: dict[str, list[np.ndarray]],
    cell: np.ndarray,
) -> tuple[str, float]:
    """Return nearest named site in xy and its xy distance in Angstrom."""
    best_name = "unknown"
    best_dist = float("inf")
    for name, positions in site_map.items():
        for pos in positions:
            dist = float(np.linalg.norm(
                _xy_delta_minimum_image(anchor_position[:2], pos[:2], cell)
            ))
            if dist < best_dist:
                best_name = name
                best_dist = dist
    return best_name, best_dist


def _classification_site_map(
    host: str,
    slab: ase.Atoms,
) -> dict[str, list[np.ndarray]]:
    if host in {"Cu(111)", "Pd(111)"}:
        return fcc111_site_candidates(slab)
    return sites_for_host(host, slab)


def _binding_atom_offset(ads_atoms: ase.Atoms, adsorbate: str) -> int:
    """Pick the adsorbate atom closest to the surface.

    The closest atom is the geometric contact point after relaxation. For
    methanol, hydrogens can sit slightly lower during tilted starts, so
    the classification prefers heavy atoms when available.
    """
    symbols = np.array(ads_atoms.get_chemical_symbols())
    z = ads_atoms.positions[:, 2]

    if adsorbate == "CH3OH":
        heavy = np.where(np.isin(symbols, ["C", "O"]))[0]
        return int(heavy[np.argmin(z[heavy])])
    if adsorbate == "CO":
        heavy = np.where(np.isin(symbols, ["C", "O"]))[0]
        return int(heavy[np.argmin(z[heavy])])
    if adsorbate == "NH3":
        # NH3 is a lone-pair donor in the relevant low-energy states; use
        # the closest atom geometrically so H-down failed starts are visible.
        return int(np.argmin(z))
    return int(np.argmin(z))


def _tilt_angle_deg(ads_atoms: ase.Atoms, adsorbate: str) -> float | None:
    """Return a simple molecule-axis tilt relative to surface normal.

    0 degrees means upright along z; 90 degrees means flat. The sign of
    the vector is ignored because site classification already records
    the binding atom.
    """
    symbols = ads_atoms.get_chemical_symbols()
    positions = ads_atoms.positions

    vec = None
    if adsorbate == "CO" and "C" in symbols and "O" in symbols:
        vec = positions[symbols.index("O")] - positions[symbols.index("C")]
    elif adsorbate == "H2O" and "O" in symbols:
        o_idx = symbols.index("O")
        h_idx = [i for i, s in enumerate(symbols) if s == "H"]
        if h_idx:
            vec = positions[h_idx].mean(axis=0) - positions[o_idx]
    elif adsorbate == "CH3OH" and "C" in symbols and "O" in symbols:
        vec = positions[symbols.index("C")] - positions[symbols.index("O")]
    elif adsorbate == "NH3" and "N" in symbols:
        n_idx = symbols.index("N")
        h_idx = [i for i, s in enumerate(symbols) if s == "H"]
        if h_idx:
            vec = positions[h_idx].mean(axis=0) - positions[n_idx]

    if vec is None:
        return None

    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return None
    cosang = abs(float(vec[2]) / norm)
    cosang = min(1.0, max(-1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def _max_force_ev_A(opt: OptimizationResult) -> float:
    """Return max force magnitude, or NaN if force data are unavailable."""
    forces = np.asarray(opt.forces, dtype=float)
    if forces.size == 0:
        return float("nan")
    forces = forces.reshape(-1, 3)
    return float(np.max(np.linalg.norm(forces, axis=1)))


def _adsorbate_integrity_status(
    initial_atoms: ase.Atoms,
    final_atoms: ase.Atoms,
    slab_atom_count: int,
    desorption_height_A: float = 5.0,
) -> str:
    """Classify obvious failed adsorbate geometries before energy ranking."""
    from ase.data import covalent_radii

    initial_ads = initial_atoms[slab_atom_count:]
    final_slab = final_atoms[:slab_atom_count]
    final_ads = final_atoms[slab_atom_count:]

    if len(final_ads) == 0:
        return "missing_adsorbate"

    top_z = float(final_slab.positions[:, 2].max())
    min_ads_z = float(final_ads.positions[:, 2].min())
    if min_ads_z - top_z > desorption_height_A:
        return "desorbed"

    if len(final_ads) > 1 and len(initial_ads) == len(final_ads):
        numbers = initial_ads.numbers
        bonded_pairs: list[tuple[int, int, float]] = []
        for i in range(len(initial_ads)):
            for j in range(i + 1, len(initial_ads)):
                d_initial = float(np.linalg.norm(
                    initial_ads.positions[i] - initial_ads.positions[j]
                ))
                bond_cutoff = 1.25 * (
                    float(covalent_radii[numbers[i]])
                    + float(covalent_radii[numbers[j]])
                )
                if 0.1 < d_initial <= bond_cutoff:
                    bonded_pairs.append((i, j, d_initial))

        for i, j, d_initial in bonded_pairs:
            d_final = float(np.linalg.norm(
                final_ads.positions[i] - final_ads.positions[j]
            ))
            if d_final > 1.5 * d_initial:
                return "dissociated"

    return "adsorbed"


def classify_final_site(
    host: str,
    adsorbate: str,
    final_atoms: ase.Atoms,
    config: Configuration,
    slab_atom_count: int,
) -> FinalSiteAnalysis:
    """Classify the final relaxed site for one configuration."""
    slab = final_atoms[:slab_atom_count]
    ads = final_atoms[slab_atom_count:]
    if len(ads) == 0:
        raise ValueError("final_atoms contains no adsorbate atoms after slab_atom_count")

    offset = _binding_atom_offset(ads, adsorbate)
    binding_position = ads.positions[offset]
    site_map = _classification_site_map(host, slab)
    final_site, site_dist = _nearest_site_name(binding_position, site_map, slab.cell.array)

    slab_positions = slab.positions
    top_z = float(slab_positions[:, 2].max())
    binding_height = float(binding_position[2] - top_z)

    distances = np.linalg.norm(slab_positions - binding_position, axis=1)
    nearest_surface_distance = float(distances.min())

    return FinalSiteAnalysis(
        label=config.label,
        host=host,
        adsorbate=adsorbate,
        start_site=config.site,
        start_orientation=config.orientation,
        final_site=final_site,
        final_site_distance_A=site_dist,
        binding_atom_symbol=ads.get_chemical_symbols()[offset],
        binding_atom_index=slab_atom_count + offset,
        binding_height_A=binding_height,
        nearest_surface_distance_A=nearest_surface_distance,
        tilt_deg=_tilt_angle_deg(ads, adsorbate),
    )


def reference_site_matches(final_site: str, ref_site: str | None) -> bool | None:
    """Conservative text match between geometric site and reference site."""
    if ref_site is None:
        return None
    final = final_site.lower()
    ref = ref_site.lower().replace("_", "-")
    aliases = {
        "top": ["top", "atop"],
        "bridge": ["bridge"],
        "fcc": ["fcc", "fcc-hollow"],
        "hcp": ["hcp", "hcp-hollow"],
        "al-top": ["al-top", "al top", "aluminium top", "aluminum top"],
        "o-top": ["o-top", "o top", "oxygen top"],
        "hollow": ["hollow"],
    }
    return any(token in ref for token in aliases.get(final, [final]))


def build_pair_results_table(
    host: str,
    adsorbate: str,
    configs: list[Configuration],
    opt_results: list[OptimizationResult],
    clean_slab_atoms: ase.Atoms,
    e_clean_slab_ev: float,
    e_gas_ads_ev: float,
    backend: str | None = None,
    execution_path: str | None = None,
    reliability_max_force_eV_A: float = 0.2,
    desorption_height_A: float = 5.0,
) -> pd.DataFrame:
    """Build one per-configuration DataFrame for a relaxed pair."""
    if execution_path is None:
        execution_path = backend or "unknown"
    if len(configs) != len(opt_results):
        raise ValueError(
            "Relaxation result count does not match configuration count: "
            f"{len(opt_results)} results for {len(configs)} configs"
        )

    rows = []
    slab_n = len(clean_slab_atoms)
    for config, opt in zip(configs, opt_results):
        final_atoms = atomic_data_to_ase(opt)
        site = classify_final_site(host, adsorbate, final_atoms, config, slab_n)
        e_ads = compute_adsorption_energy_ev(
            float(opt.energy), e_clean_slab_ev, e_gas_ads_ev
        )
        max_force = _max_force_ev_A(opt)
        geometry_status = _adsorbate_integrity_status(
            config.atoms,
            final_atoms,
            slab_atom_count=slab_n,
            desorption_height_A=desorption_height_A,
        )
        force_ok = bool(np.isnan(max_force) or max_force <= reliability_max_force_eV_A)
        reliable_for_minimum = bool(
            opt.converged
            and np.isfinite(e_ads)
            and force_ok
            and geometry_status == "adsorbed"
        )
        row = {
            "backend": execution_path,
            "pair": f"{adsorbate}/{host}",
            "label": config.label,
            "start_site": config.site,
            "start_orientation": config.orientation,
            "rot_deg": config.rot_deg,
            "height_A_start": config.height,
            "converged": opt.converged,
            "optimizer_nsteps": opt.num_optimization_steps,
            "max_force_eV_A": max_force,
            "geometry_status": geometry_status,
            "reliable_for_minimum": reliable_for_minimum,
            "E_ads_eV": e_ads,
            "E_ads (eV)": e_ads,
            # Backward-compatible aliases for older cached outputs and reports.
            "E_bind_eV": e_ads,
            "E_bind (eV)": e_ads,
            "reference_scope": "none",
            "validation_status": "not-evaluated",
        }
        site_record = asdict(site)
        if geometry_status != "adsorbed":
            site_record["tilt_deg"] = None
        row.update(site_record)
        rows.append(row)
    return pd.DataFrame(rows)


def _validation_status(
    e_mace_ev: float,
    ref: AdsorbMLReference | None,
    site_match: bool | None,
    mad_ev: float,
) -> tuple[str, float | None, float | None]:
    if ref is None or ref.e_ads_ev is None:
        return "no-reference", None, None

    delta = float(e_mace_ev) - float(ref.e_ads_ev)
    scaled = abs(delta) / mad_ev
    if not ref.strict_for_parity:
        return "context-only", delta, scaled
    if site_match is False:
        return "site-mismatch", delta, scaled
    if abs(delta) <= mad_ev:
        return "pass", delta, scaled
    if abs(delta) <= 2 * mad_ev:
        return "energy-warning", delta, scaled
    return "energy-fail", delta, scaled


def summarize_pair_validation(
    pair_results: dict[tuple[str, str], pd.DataFrame],
    references: dict[tuple[str, str], AdsorbMLReference],
    mad_ev: float = LITERATURE_OC157_MAD_GUIDE_EV,
) -> pd.DataFrame:
    """Return one validation row per pair using the batch-minimum structure."""
    rows: list[ValidationResult] = []
    for (host, adsorbate), df in pair_results.items():
        if df.empty:
            continue
        candidates = df
        if "reliable_for_minimum" in df.columns:
            candidates = df[df["reliable_for_minimum"]]
        e_col = adsorption_energy_column(df)
        if candidates.empty:
            winner = df.loc[df[e_col].idxmin()]
            winner_is_reliable = False
        else:
            winner = candidates.loc[candidates[e_col].idxmin()]
            winner_is_reliable = True
        ref = references.get((host, adsorbate))
        site_match = reference_site_matches(
            str(winner["final_site"]), ref.binding_site if ref else None
        )
        if winner_is_reliable:
            status, delta, scaled = _validation_status(
                float(winner[e_col]), ref, site_match, mad_ev
            )
            notes = ref.notes if ref else ""
        else:
            status, delta, scaled = "no-reliable-result", None, None
            notes = "No converged adsorbed configuration passed the reliability filter."
        rows.append(ValidationResult(
            host=host,
            adsorbate=adsorbate,
            pair=f"{adsorbate}/{host}",
            tier=ref.tier if ref else "?",
            reference_scope=ref.reference_scope if ref else "none",
            MACE_site=f"{winner['final_site']} ({winner['binding_atom_symbol']}-down)",
            reference_site=ref.binding_site if ref else "-",
            site_match=site_match,
            E_MACE_eV=round(float(winner[e_col]), 3),
            E_ref_eV=round(ref.e_ads_ev, 3) if ref and ref.e_ads_ev is not None else None,
            delta_E_eV=round(delta, 3) if delta is not None else None,
            abs_delta_over_MAD=round(scaled, 2) if scaled is not None else None,
            status=status,
            notes=notes,
        ))
    return pd.DataFrame([asdict(row) for row in rows])


def strict_parity_subset(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Rows approved for strict MACE-vs-DFT parity statistics."""
    if summary_df.empty:
        return summary_df
    return summary_df[summary_df["reference_scope"].isin(["strict", "near-strict"])].copy()
