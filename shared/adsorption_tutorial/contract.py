"""Backend-neutral scientific contract for the adsorption tutorial.

The BGR NIM notebook and the future toolkit notebook should import or mirror
this contract rather than redefining the active panel independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BackendName = Literal["bgr_nim", "toolkit"]


@dataclass(frozen=True)
class HostSpec:
    name: str
    material_class: str
    facet: str
    bgr_helper: str
    toolkit_helper: str | None
    slab_layers: int | None
    surface_sites_per_cell: int | None
    default_supercell: tuple[int, int, int]
    vacuum_A: float


@dataclass(frozen=True)
class AdsorbateSpec:
    name: str
    active: bool
    orientations: tuple[str, ...]
    role: str


@dataclass(frozen=True)
class ResultColumn:
    name: str
    units: str | None
    description: str


ACTIVE_HOSTS: tuple[HostSpec, ...] = (
    HostSpec(
        name="Cu(111)",
        material_class="fcc metal",
        facet="(111)",
        bgr_helper="helpers.build_cu111_slab",
        toolkit_helper=None,
        slab_layers=4,
        surface_sites_per_cell=9,
        default_supercell=(3, 3, 1),
        vacuum_A=15.0,
    ),
    HostSpec(
        name="Pd(111)",
        material_class="fcc metal",
        facet="(111)",
        bgr_helper="helpers.build_pd111_slab",
        toolkit_helper=None,
        slab_layers=4,
        surface_sites_per_cell=9,
        default_supercell=(3, 3, 1),
        vacuum_A=15.0,
    ),
    HostSpec(
        name="Al2O3(0001)",
        material_class="oxide support",
        facet="alpha-Al2O3(0001)",
        bgr_helper="helpers.build_alpha_alumina_0001_slab",
        toolkit_helper=None,
        slab_layers=None,
        surface_sites_per_cell=None,
        default_supercell=(1, 1, 1),
        vacuum_A=15.0,
    ),
)


ACTIVE_ADSORBATES: tuple[AdsorbateSpec, ...] = (
    AdsorbateSpec(
        name="CO",
        active=True,
        orientations=("C-down", "O-down"),
        role="active panel",
    ),
    AdsorbateSpec(
        name="H2O",
        active=True,
        orientations=("O-down", "H-down", "flat"),
        role="active panel",
    ),
    AdsorbateSpec(
        name="CH3OH",
        active=True,
        orientations=("O-down", "methyl-down"),
        role="active panel",
    ),
)


OPTIONAL_ADSORBATES: tuple[AdsorbateSpec, ...] = (
    AdsorbateSpec(
        name="NH3",
        active=False,
        orientations=("N-down", "H-down", "flat"),
        role="optional first-binding context only",
    ),
)


ACTIVE_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (host.name, adsorbate.name)
    for host in ACTIVE_HOSTS
    for adsorbate in ACTIVE_ADSORBATES
)


REQUIRED_RESULT_COLUMNS: tuple[ResultColumn, ...] = (
    ResultColumn("backend", None, "Execution backend: bgr_nim or toolkit."),
    ResultColumn("host", None, "Surface host name from ACTIVE_HOSTS."),
    ResultColumn("adsorbate", None, "Adsorbate name from ACTIVE_ADSORBATES."),
    ResultColumn("label", None, "Unique configuration label."),
    ResultColumn("start_site", None, "Starting adsorption-site class."),
    ResultColumn("start_orientation", None, "Starting adsorbate orientation."),
    ResultColumn("final_site", None, "Relaxed final adsorption-site class."),
    ResultColumn("E_bind_eV", "eV", "Adsorption energy; negative is exothermic."),
    ResultColumn("converged", None, "Optimizer convergence flag."),
    ResultColumn("max_force_eV_A", "eV/A", "Maximum final force magnitude."),
    ResultColumn("geometry_status", None, "adsorbed, desorbed, or dissociated."),
    ResultColumn(
        "reliable_for_minimum",
        None,
        "Whether this row is eligible for batch-minimum selection.",
    ),
    ResultColumn("reference_scope", None, "context, near-strict, strict, or none."),
    ResultColumn("validation_status", None, "Reference-aware validation label."),
)
