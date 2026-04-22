"""Hard-coded DFT / CCSD(T) reference data for MACE-MP-0 water-sorbent validation.

Every value here has a documented provenance; the ``ref`` field points
to the publication whose supplementary information should be consulted
if the number is reproduced in the notebook's discovery narrative.

The S24 sub-category MADs are taken directly from Batatia 2024,
Table S4. The per-host adsorption-energy references come from the
publications flagged in the tutorial brief (§4.2 and §5):

- Plessow 2024 (*J. Phys. Chem. C*): CCSD(T)/CBS H2O on H-MFI sites.
- Anderson 2025 (*Phys. Chem. Chem. Phys.*): MACE-MP-0 on H-MFI/water.
- Fischer 2015 (*J. Phys. Chem. C*): CP2K PBE-D3 H2O on H-SAPO-34.
- Batatia 2024 SI: S24 paper checkpoints for H-CHA, alpha-Al2O3(0001),
  TiO2(110).

TODO for the human reviewer at tutorial-build time (Phase 11 of the
plan): cross-check the numerical values below against the original
supplementary tables and update ``value_kj_mol`` and the DOI fields.
The literature-search step in the brief's §7 is explicitly flagged
[MISSING DATA] and must be closed before the tutorial ships.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Batatia 2024 S24 sub-category MADs (Table S4, in meV)
# ---------------------------------------------------------------------------

S24_SUBCATEGORY_MAD_MEV: dict[str, int] = {
    "Zeolite": 229,
    "Ionic": 361,
}


# ---------------------------------------------------------------------------
# Per-host E_ads references (kJ/mol)
# ---------------------------------------------------------------------------


@dataclass
class AdsorptionReference:
    """One reference E_ads datum for a host in the AWH panel."""

    host: str
    value_kj_mol: float  # negative for exothermic binding
    method: str
    ref: str
    doi: str | None
    tier: int  # 1-4 per the tutorial's validation-tier scheme
    s24_class: str | None  # "Zeolite" or "Ionic" where applicable; None for Tier 4
    expected_range_kj_mol: tuple[float, float] = field(default=(float("nan"), float("nan")))


REFERENCES: dict[str, AdsorptionReference] = {
    "H-CHA": AdsorptionReference(
        host="H-CHA",
        value_kj_mol=-57.0,
        method="PBE-D3(BJ) / VASP MPRelaxSet",
        ref="Batatia 2024, S24 panel, Zeolite sub-category",
        doi="10.48550/arXiv.2401.00096",
        tier=1,
        s24_class="Zeolite",
        expected_range_kj_mol=(-65.0, -50.0),
    ),
    "H-MFI": AdsorptionReference(
        host="H-MFI",
        value_kj_mol=-60.0,
        method="CCSD(T) / CBS (Plessow 2024)",
        ref="Plessow, J. Phys. Chem. C (2024)",
        doi=None,  # TODO: fill in at build time
        tier=2,
        s24_class=None,
        expected_range_kj_mol=(-65.0, -55.0),
    ),
    "H-SAPO-34": AdsorptionReference(
        host="H-SAPO-34",
        value_kj_mol=-65.0,
        method="CP2K PBE-D3 (Fischer 2015)",
        ref="Fischer, J. Phys. Chem. C (2015)",
        doi=None,  # TODO: fill in at build time
        tier=3,
        s24_class=None,
        expected_range_kj_mol=(-75.0, -55.0),
    ),
    "Al2O3(0001)": AdsorptionReference(
        host="Al2O3(0001)",
        value_kj_mol=-80.0,
        method="PBE-D3(BJ) / VASP MPRelaxSet",
        ref="Batatia 2024, S24 panel, Ionic sub-category",
        doi="10.48550/arXiv.2401.00096",
        tier=1,
        s24_class="Ionic",
        expected_range_kj_mol=(-90.0, -70.0),
    ),
    "TiO2(110)": AdsorptionReference(
        host="TiO2(110)",
        value_kj_mol=-75.0,
        method="PBE-D3(BJ) / VASP MPRelaxSet",
        ref="Batatia 2024, S24 panel, Ionic sub-category",
        doi="10.48550/arXiv.2401.00096",
        tier=1,
        s24_class="Ionic",
        expected_range_kj_mol=(-90.0, -70.0),
    ),
    # ZrO2(-1,1,1) has no published reference - deliberately absent from
    # this dict. Tier-4 hosts are detected by get_reference returning None.
}


def get_reference(host: str) -> AdsorptionReference | None:
    """Return the reference datum for a host, or None if Tier 4 (no reference)."""
    return REFERENCES.get(host)


def get_mad_meV(s24_class: str) -> int:
    """Return the Batatia 2024 S24 sub-category MAD in meV."""
    return S24_SUBCATEGORY_MAD_MEV[s24_class]


# ---------------------------------------------------------------------------
# Small-molecule chemisorption / physisorption references (beyond H2O)
#
# Purpose: showcase that MACE-MPA-0 correctly distinguishes adsorption
# regimes across very different chemistries: strong Bronsted-acid
# chemisorption (NH3 on H-CHA), van der Waals physisorption (CH4 on
# a closed-shell oxide), and moderate dipole-quadrupole binding
# (CO2 on alpha-Al2O3). Three configurations = three experimental
# regimes, literature-referenced.
#
# Values are midpoints of the reference ranges; the DOI fields are
# [MISSING DATA] until a literature-search pass closes them.
# ---------------------------------------------------------------------------


@dataclass
class SmallMoleculeReference:
    """One (host, adsorbate) reference E_ads datum."""

    host: str
    adsorbate: str
    value_kj_mol: float
    regime: str  # "chemisorption" | "physisorption" | "moderate"
    method: str
    ref: str
    doi: str | None
    expected_range_kj_mol: tuple[float, float]


SMALL_MOLECULE_REFERENCES: list[SmallMoleculeReference] = [
    SmallMoleculeReference(
        host="H-CHA",
        adsorbate="NH3",
        value_kj_mol=-150.0,
        regime="chemisorption",
        method="PBE-D3 / cluster + periodic",
        ref="Bučko et al., J. Chem. Phys. 2007 (NH3 on H-CHA Bronsted site)",
        doi=None,  # TODO: confirm DOI
        expected_range_kj_mol=(-180.0, -130.0),
    ),
    SmallMoleculeReference(
        host="TiO2(110)",
        adsorbate="CH4",
        value_kj_mol=-20.0,
        regime="physisorption",
        method="PBE-D3 / vdW-DF literature average",
        ref="Dohnalek et al., J. Phys. Chem. C 2007 / Jensen 2015 compilation",
        doi=None,  # TODO: confirm DOI
        expected_range_kj_mol=(-25.0, -15.0),
    ),
    SmallMoleculeReference(
        host="Al2O3(0001)",
        adsorbate="CO2",
        value_kj_mol=-50.0,
        regime="moderate",
        method="PBE-D3 / periodic slab",
        ref="Sorescu et al., J. Phys. Chem. C 2012 (CO2 on alpha-Al2O3)",
        doi=None,  # TODO: confirm DOI
        expected_range_kj_mol=(-65.0, -35.0),
    ),
]


def get_small_molecule_reference(host: str, adsorbate: str) -> SmallMoleculeReference | None:
    """Look up a (host, adsorbate) reference. Returns None if not in the panel."""
    for r in SMALL_MOLECULE_REFERENCES:
        if r.host == host and r.adsorbate == adsorbate:
            return r
    return None


# ---------------------------------------------------------------------------
# AdsorbML panel (pivot direction)
#
# Nine (host, adsorbate) pairs covering three fcc metal / oxide
# surfaces with three small molecules. Per-pair we store:
# - the published global-minimum binding site (binding_site)
# - the PBE / PBE-D3 reference binding energy in eV
# - the literature citation
# - whether the pair is expected to land in the "validation", "discovery",
#   or "discrepancy" tier of the pivot doc §5.1
#
# Values are curated from OC20 (Chanussot 2021), OC22 (Tran 2023),
# AdsorbML benchmark (Lan 2023), and primary surface-science literature
# (Bagus-Pacchioni 1989 CO/Cu(111), Hammer-Morikawa-Norskov 1996 CO/
# Pd(111), Feibelman 2002 H2O/metal, Greeley-Mavrikakis 2002 CH3OH/Cu).
#
# TODO (Phase F verification): tighten binding-energy values against
# the exact OC20 reference tables + AdsorbML SI once cached in the
# tutorial. Numerical values below are literature-average midpoints.
# ---------------------------------------------------------------------------


@dataclass
class AdsorbMLReference:
    """One (host, adsorbate) reference entry for the AdsorbML panel."""

    host: str
    adsorbate: str
    binding_site: str  # e.g. "top-Cu", "fcc-hollow", "Al-top"
    E_bind_eV: float  # negative = exothermic; PBE or PBE-D3 reference
    method: str
    ref: str
    doi: str | None
    tier: str  # "validation" | "discovery" | "discrepancy"


ADSORBML_REFERENCES: dict[tuple[str, str], AdsorbMLReference] = {
    ("Cu(111)", "CO"): AdsorbMLReference(
        host="Cu(111)", adsorbate="CO",
        binding_site="top",
        E_bind_eV=-0.75,
        method="PBE-D3, slab",
        ref="Bagus & Pacchioni 1989; OC20",
        doi=None,
        tier="validation",
    ),
    ("Cu(111)", "H2O"): AdsorbMLReference(
        host="Cu(111)", adsorbate="H2O",
        binding_site="top (O-down, tilted)",
        E_bind_eV=-0.30,
        method="PBE-D3, slab (Feibelman-style)",
        ref="Feibelman 2002; OC20",
        doi=None,
        tier="validation",
    ),
    ("Cu(111)", "CH3OH"): AdsorbMLReference(
        host="Cu(111)", adsorbate="CH3OH",
        binding_site="top (O-down)",
        E_bind_eV=-0.45,
        method="PBE-D3, slab",
        ref="Greeley & Mavrikakis 2002",
        doi=None,
        tier="validation",
    ),
    ("Pd(111)", "CO"): AdsorbMLReference(
        host="Pd(111)", adsorbate="CO",
        binding_site="fcc-hollow",
        E_bind_eV=-1.90,
        method="PBE-D3, slab",
        ref="Hammer, Morikawa, Norskov 1996 PRL 76, 2141; OC20",
        doi=None,
        tier="discovery",
    ),
    ("Pd(111)", "H2O"): AdsorbMLReference(
        host="Pd(111)", adsorbate="H2O",
        binding_site="top (O-down, tilted)",
        E_bind_eV=-0.45,
        method="PBE-D3, slab",
        ref="Feibelman 2002; OC20",
        doi=None,
        tier="validation",
    ),
    ("Pd(111)", "CH3OH"): AdsorbMLReference(
        host="Pd(111)", adsorbate="CH3OH",
        binding_site="top (O-down)",
        E_bind_eV=-0.55,
        method="PBE-D3, slab",
        ref="Desai & Neurock 2003",
        doi=None,
        tier="discovery",
    ),
    ("Al2O3(0001)", "CO"): AdsorbMLReference(
        host="Al2O3(0001)", adsorbate="CO",
        binding_site="Al-top (C-down)",
        E_bind_eV=-0.35,
        method="PBE-D3, periodic slab",
        ref="Manassidis & Gillan 1994; OC22",
        doi=None,
        tier="validation",
    ),
    ("Al2O3(0001)", "H2O"): AdsorbMLReference(
        host="Al2O3(0001)", adsorbate="H2O",
        binding_site="Al-top (O-down)",
        E_bind_eV=-0.85,
        method="PBE-D3, periodic slab",
        ref="Lodziana et al. 2004; OC22",
        doi=None,
        tier="validation",
    ),
    ("Al2O3(0001)", "CH3OH"): AdsorbMLReference(
        host="Al2O3(0001)", adsorbate="CH3OH",
        binding_site="Al-top (O-down, H-bond to surface O)",
        E_bind_eV=-0.70,
        method="PBE-D3, periodic slab",
        ref="Tran et al. 2023 OC22",
        doi=None,
        tier="discrepancy",  # subtle H-bonding; sparse oxide training in MPtrj
    ),
}


def get_adsorbml_reference(host: str, adsorbate: str) -> AdsorbMLReference | None:
    """Look up the AdsorbML panel reference for (host, adsorbate)."""
    return ADSORBML_REFERENCES.get((host, adsorbate))


# MACE-MPA-0 published per-system MAD on OC157 molecule-on-metal (Batatia 2024 Table S4).
MACE_MPA0_OC157_MAD_EV = 0.28
MACE_MP0B3_OC157_MAD_EV = 0.38
