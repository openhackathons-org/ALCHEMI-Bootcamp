"""Scientific contract for the adsorption tutorial.

The Toolkit teaching notebook should import or mirror this contract rather than
redefining the active panel independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BackendName = Literal["toolkit"]


@dataclass(frozen=True)
class HostSpec:
    name: str
    material_class: str
    facet: str
    structure_helper: str
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
        structure_helper="helpers.build_cu111_slab",
        toolkit_helper=None,
        slab_layers=4,
        surface_sites_per_cell=9,
        default_supercell=(3, 3, 1),
        vacuum_A=15.0,
    ),
    HostSpec(
        name="Cu(100)",
        material_class="fcc metal",
        facet="(100)",
        structure_helper="helpers.build_cu100_slab",
        toolkit_helper=None,
        slab_layers=4,
        surface_sites_per_cell=9,
        default_supercell=(3, 3, 1),
        vacuum_A=15.0,
    ),
    HostSpec(
        name="Cu(110)",
        material_class="fcc metal",
        facet="(110)",
        structure_helper="helpers.build_cu110_slab",
        toolkit_helper=None,
        slab_layers=4,
        surface_sites_per_cell=9,
        default_supercell=(3, 3, 1),
        vacuum_A=15.0,
    ),
    HostSpec(
        name="TiO2(110)",
        material_class="oxide",
        facet="rutile TiO2(110)",
        structure_helper="helpers.build_tio2_110_slab",
        toolkit_helper=None,
        slab_layers=None,
        surface_sites_per_cell=None,
        default_supercell=(2, 2, 1),
        vacuum_A=15.0,
    ),
    HostSpec(
        name="TiO2(100)",
        material_class="oxide",
        facet="rutile TiO2(100)",
        structure_helper="helpers.build_tio2_100_slab",
        toolkit_helper=None,
        slab_layers=None,
        surface_sites_per_cell=None,
        default_supercell=(2, 2, 1),
        vacuum_A=15.0,
    ),
    HostSpec(
        name="TiO2(101)",
        material_class="oxide",
        facet="rutile TiO2(101)",
        structure_helper="helpers.build_tio2_101_slab",
        toolkit_helper=None,
        slab_layers=None,
        surface_sites_per_cell=None,
        default_supercell=(2, 2, 1),
        vacuum_A=15.0,
    ),
    HostSpec(
        name="TiN(001)",
        material_class="nitride ceramic",
        facet="rocksalt TiN(001)",
        structure_helper="helpers.build_tin_001_slab",
        toolkit_helper=None,
        slab_layers=None,
        surface_sites_per_cell=None,
        default_supercell=(2, 2, 1),
        vacuum_A=15.0,
    ),
    HostSpec(
        name="TiN(110)",
        material_class="nitride ceramic",
        facet="rocksalt TiN(110)",
        structure_helper="helpers.build_tin_110_slab",
        toolkit_helper=None,
        slab_layers=None,
        surface_sites_per_cell=None,
        default_supercell=(2, 2, 1),
        vacuum_A=15.0,
    ),
    HostSpec(
        name="TiN(210)",
        material_class="nitride ceramic",
        facet="rocksalt TiN(210)",
        structure_helper="helpers.build_tin_210_slab",
        toolkit_helper=None,
        slab_layers=None,
        surface_sites_per_cell=None,
        default_supercell=(2, 2, 1),
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
        orientations=("O-down", "H-down"),
        role="active panel",
    ),
    AdsorbateSpec(
        name="NH3",
        active=True,
        orientations=("N-down", "H-down"),
        role="active panel",
    ),
    AdsorbateSpec(
        name="CH3OH",
        active=True,
        orientations=("O-down", "methyl-down"),
        role="active panel",
    ),
)


VALIDATION_ADSORBATES: tuple[AdsorbateSpec, ...] = (
    AdsorbateSpec(
        name="NH3",
        active=False,
        orientations=("dataset-provided",),
        role="OC20Dense closed-shell benchmark slice; not active panel",
    ),
    AdsorbateSpec(
        name="N2",
        active=False,
        orientations=("dataset-provided",),
        role="OC20Dense closed-shell benchmark slice; not active panel",
    ),
)


OPTIONAL_ADSORBATES: tuple[AdsorbateSpec, ...] = VALIDATION_ADSORBATES


ACTIVE_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (host.name, adsorbate.name)
    for host in ACTIVE_HOSTS
    for adsorbate in ACTIVE_ADSORBATES
)


REQUIRED_RESULT_COLUMNS: tuple[ResultColumn, ...] = (
    ResultColumn("backend", None, "Execution path, currently toolkit."),
    ResultColumn("host", None, "Surface host name from ACTIVE_HOSTS."),
    ResultColumn("adsorbate", None, "Adsorbate name from ACTIVE_ADSORBATES."),
    ResultColumn("label", None, "Unique configuration label."),
    ResultColumn("start_site", None, "Starting adsorption-site class."),
    ResultColumn("start_orientation", None, "Starting adsorbate orientation."),
    ResultColumn("final_site", None, "Relaxed final adsorption-site class."),
    ResultColumn("E_ads_eV", "eV", "Canonical adsorption energy; negative is exothermic."),
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
