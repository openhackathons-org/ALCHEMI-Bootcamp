# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reference data for the AdsorbML MACE-vs-DFT tutorial.

The active tutorial uses ``ADSORBML_REFERENCES`` and
``REFERENCE_PUBLICATIONS`` below. Older AWH/S24 data structures remain in
this file for auxiliary notebooks and tests, but should not drive the
current AdsorbML notebook.

Every value here has a documented provenance; the ``ref`` field points to
the publication whose supplementary information should be consulted if
the number is reproduced in the notebook's discovery narrative.

Legacy S24/AWH notes:

- Plessow 2024 (*J. Phys. Chem. C*): CCSD(T)/CBS H2O on H-MFI sites.
- Anderson 2025 (*Phys. Chem. Chem. Phys.*): MACE-MP-0 on H-MFI/water.
- Fischer 2015 (*J. Phys. Chem. C*): CP2K PBE-D3 H2O on H-SAPO-34.
- Batatia 2024 SI: S24 paper checkpoints for H-CHA, alpha-Al2O3(0001),
  TiO2(110).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Batatia 2024 S24 sub-category MADs (Table S4, in meV)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferencePublication:
    """Canonical publication metadata used by tutorial markdown and tables."""

    key: str
    citation: str
    doi: str | None
    url: str
    role: str


REFERENCE_PUBLICATIONS: dict[str, ReferencePublication] = {
    "mace_mp0": ReferencePublication(
        key="mace_mp0",
        citation="Batatia et al. 2024, A foundation model for atomistic materials chemistry",
        doi="10.48550/arXiv.2401.00096",
        url="https://arxiv.org/abs/2401.00096",
        role="Published model-level OC157 relative-energy MAD values; orientation only, not tutorial-run error bars.",
    ),
    "adsorbml": ReferencePublication(
        key="adsorbml",
        citation="Lan et al. 2023, AdsorbML: a leap in efficiency for adsorption energy calculations using generalizable machine learning potentials",
        doi="10.1038/s41524-023-01121-5",
        url="https://doi.org/10.1038/s41524-023-01121-5",
        role="Configuration-search methodology; single-start vs batch-search reliability.",
    ),
    "oc20": ReferencePublication(
        key="oc20",
        citation="Chanussot et al. 2021, Open Catalyst 2020 (OC20) Dataset and Community Challenges",
        doi="10.1021/acscatal.0c04525",
        url="https://doi.org/10.1021/acscatal.0c04525",
        role="DFT reference source for metal-surface adsorption records.",
    ),
    "oc22": ReferencePublication(
        key="oc22",
        citation="Tran et al. 2023, The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysts",
        doi="10.1021/acscatal.2c05426",
        url="https://doi.org/10.1021/acscatal.2c05426",
        role="DFT reference source for oxide-surface adsorption records.",
    ),
    "nh3_cu111": ReferencePublication(
        key="nh3_cu111",
        citation="Mayers et al. 2002, The local adsorption geometry of CH3 and NH3 on Cu(111): a density functional theory study",
        doi="10.1016/S0039-6028(01)01769-1",
        url="https://doi.org/10.1016/S0039-6028(01)01769-1",
        role="NH3/Cu(111) atop-site and adsorption-energy context.",
    ),
    "nh3_pd111": ReferencePublication(
        key="nh3_pd111",
        citation="Herron, Tonelli, and Mavrikakis 2012, Atomic and Molecular Adsorption on Pd(111)",
        doi="10.1016/j.susc.2012.07.003",
        url="https://doi.org/10.1016/j.susc.2012.07.003",
        role="NH3/Pd(111) molecular adsorption and decomposition context.",
    ),
    "nh3_alumina": ReferencePublication(
        key="nh3_alumina",
        citation="Kelber 2007, Alumina surfaces and interfaces under non-ultrahigh vacuum conditions",
        doi="10.1016/j.surfrep.2006.12.003",
        url="https://doi.org/10.1016/j.surfrep.2006.12.003",
        role="NH3/alumina support chemistry and Lewis-acid-site context.",
    ),
    "nh3_scr_alumina": ReferencePublication(
        key="nh3_scr_alumina",
        citation="McBriarty and Ellis 2016, Cation synergies affect ammonia adsorption over VOx and (V,W)Ox dispersed on alpha-Al2O3(0001) and alpha-Fe2O3(0001)",
        doi="10.1016/j.susc.2016.03.015",
        url="https://doi.org/10.1016/j.susc.2016.03.015",
        role="NH3 adsorption relevance for oxide-supported NO-SCR chemistry.",
    ),
}

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
    expected_range_kj_mol: tuple[float, float] = field(
        default=(float("nan"), float("nan"))
    )


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


def get_small_molecule_reference(
    host: str, adsorbate: str
) -> SmallMoleculeReference | None:
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
#   or "discrepancy" tier of the pivot doc Section 5.1
# - whether the value is strict enough for apples-to-apples parity
#
# Values are curated from OC20 (Chanussot 2021), OC22 (Tran 2023),
# AdsorbML benchmark (Lan 2023), and primary surface-science literature
# (Bagus-Pacchioni 1989 CO/Cu(111), Hammer-Morikawa-Norskov 1996 CO/
# Pd(111), Feibelman 2002 H2O/metal, Greeley-Mavrikakis 2002 CH3OH/Cu).
#
# IMPORTANT: reference_scope controls whether a row may be used for the
# strict parity plot. Most values below remain "context" until the exact
# OC20/OC22/AdsorbML records are cached into the tutorial with matching
# slab, coverage, functional, and sign convention. Context rows are still
# scientifically useful, but the notebook must label them as contextual.
# ---------------------------------------------------------------------------


@dataclass
class AdsorbMLReference:
    """One (host, adsorbate) reference entry for the AdsorbML panel."""

    host: str
    adsorbate: str
    binding_site: str  # e.g. "top-Cu", "fcc-hollow", "Al-top"
    E_ads_eV: (
        float | None
    )  # negative = exothermic; PBE/PBE-D3 reference, None = site-only context
    method: str
    ref: str
    doi: str | None
    tier: str  # "validation" | "discovery" | "discrepancy"
    facet: str
    slab_layers: int | None
    supercell: str
    coverage: str
    frozen_layers: str
    functional: str
    dispersion: str
    sign_convention: str = (
        "E_ads = E_slab+ads - E_clean_slab - E_gas_ads; negative exothermic"
    )
    source_type: str = "literature"
    source_url: str | None = None
    reference_scope: str = "context"  # "strict" | "near-strict" | "context"
    confidence: str = "needs_exact_dataset_lookup"
    notes: str = ""

    @property
    def strict_for_parity(self) -> bool:
        """True when the row can be used in the strict MACE-vs-DFT parity plot."""
        return self.E_ads_eV is not None and self.reference_scope in {
            "strict",
            "near-strict",
        }

    @property
    def e_ads_ev(self) -> float | None:
        """Lowercase alias used by newer analysis code."""
        return self.E_ads_eV


ADSORBML_REFERENCES: dict[tuple[str, str], AdsorbMLReference] = {
    ("Cu(111)", "CO"): AdsorbMLReference(
        host="Cu(111)",
        adsorbate="CO",
        binding_site="top",
        E_ads_eV=-0.75,
        method="PBE-D3, slab",
        ref="Bagus & Pacchioni 1989; OC20",
        doi="10.1021/acscatal.0c04525",
        tier="validation",
        facet="fcc(111)",
        slab_layers=4,
        supercell="3x3 tutorial slab; verify against OC20 row before strict parity",
        coverage="1 adsorbate per 3x3 surface cell in tutorial",
        frozen_layers="bottom 2 layers fixed in tutorial",
        functional="PBE family",
        dispersion="D3 in tutorial; OC20 reference may be RPBE/PBE without D3 depending on row",
        source_type="OC20 + primary literature context",
        source_url="https://doi.org/10.1021/acscatal.0c04525",
        reference_scope="context",
        notes="Top-site preference is the site checkpoint; exact energy must be replaced by matching OC20/AdsorbML datum before strict parity.",
    ),
    ("Cu(111)", "H2O"): AdsorbMLReference(
        host="Cu(111)",
        adsorbate="H2O",
        binding_site="top (O-down, tilted)",
        E_ads_eV=-0.30,
        method="PBE-D3, slab (Feibelman-style)",
        ref="Feibelman 2002; OC20",
        doi=None,
        tier="validation",
        facet="fcc(111)",
        slab_layers=4,
        supercell="3x3 tutorial slab; verify against OC20 row before strict parity",
        coverage="1 adsorbate per 3x3 surface cell in tutorial",
        frozen_layers="bottom 2 layers fixed in tutorial",
        functional="PBE family",
        dispersion="D3 in tutorial; reference dispersion treatment must be verified",
        source_type="OC20 + surface-science context",
        source_url="https://doi.org/10.1021/acscatal.0c04525",
        reference_scope="context",
        notes="Water adsorption references are sensitive to coverage and clustering; single-molecule low-coverage row needed for strict parity.",
    ),
    ("Cu(111)", "CH3OH"): AdsorbMLReference(
        host="Cu(111)",
        adsorbate="CH3OH",
        binding_site="top (O-down)",
        E_ads_eV=-0.45,
        method="PBE-D3, slab",
        ref="Greeley & Mavrikakis 2002",
        doi=None,
        tier="validation",
        facet="fcc(111)",
        slab_layers=4,
        supercell="3x3 tutorial slab; verify against source before strict parity",
        coverage="1 adsorbate per 3x3 surface cell in tutorial",
        frozen_layers="bottom 2 layers fixed in tutorial",
        functional="PBE family",
        dispersion="D3 in tutorial; reference dispersion treatment must be verified",
        source_type="primary literature context",
        source_url=None,
        reference_scope="context",
        notes="Use as binding-mode context until the exact low-coverage slab value is verified.",
    ),
    ("Cu(111)", "NH3"): AdsorbMLReference(
        host="Cu(111)",
        adsorbate="NH3",
        binding_site="top (N-down)",
        E_ads_eV=-0.70,
        method="GGA DFT, slab",
        ref="Mayers et al. 2002 Surface Science; OC20 context",
        doi="10.1016/S0039-6028(01)01769-1",
        tier="validation",
        facet="fcc(111)",
        slab_layers=4,
        supercell="3x3 tutorial slab; literature row uses its own coverage/slab convention",
        coverage="1 adsorbate per 3x3 surface cell in tutorial",
        frozen_layers="bottom 2 layers fixed in tutorial",
        functional="GGA/PBE family",
        dispersion="D3 in tutorial; source value predates routine D3 correction",
        source_type="primary literature context",
        source_url="https://doi.org/10.1016/S0039-6028(01)01769-1",
        reference_scope="context",
        notes="Strong relevance case: NH3 binds through the N lone pair at Cu atop sites; source reports roughly 0.7 eV adsorption, but exact slab/coverage differs from the tutorial.",
    ),
    ("Pd(111)", "CO"): AdsorbMLReference(
        host="Pd(111)",
        adsorbate="CO",
        binding_site="fcc-hollow",
        E_ads_eV=-1.90,
        method="PBE-D3, slab",
        ref="Hammer, Morikawa, Norskov 1996 PRL 76, 2141; OC20",
        doi=None,
        tier="discovery",
        facet="fcc(111)",
        slab_layers=4,
        supercell="3x3 tutorial slab; verify against OC20 row before strict parity",
        coverage="1 adsorbate per 3x3 surface cell in tutorial",
        frozen_layers="bottom 2 layers fixed in tutorial",
        functional="PBE family",
        dispersion="D3 in tutorial; historical references often lack D3",
        source_type="OC20 + primary literature context",
        source_url="https://doi.org/10.1021/acscatal.0c04525",
        reference_scope="context",
        notes="Primary pedagogical site-bias case: top-only starts should miss the hollow minimum.",
    ),
    ("Pd(111)", "H2O"): AdsorbMLReference(
        host="Pd(111)",
        adsorbate="H2O",
        binding_site="top (O-down, tilted)",
        E_ads_eV=-0.45,
        method="PBE-D3, slab",
        ref="Feibelman 2002; OC20",
        doi=None,
        tier="validation",
        facet="fcc(111)",
        slab_layers=4,
        supercell="3x3 tutorial slab; verify against OC20 row before strict parity",
        coverage="1 adsorbate per 3x3 surface cell in tutorial",
        frozen_layers="bottom 2 layers fixed in tutorial",
        functional="PBE family",
        dispersion="D3 in tutorial; reference dispersion treatment must be verified",
        source_type="OC20 + surface-science context",
        source_url="https://doi.org/10.1021/acscatal.0c04525",
        reference_scope="context",
        notes="Single-water adsorption should be separated from bilayer/cluster references.",
    ),
    ("Pd(111)", "CH3OH"): AdsorbMLReference(
        host="Pd(111)",
        adsorbate="CH3OH",
        binding_site="top (O-down)",
        E_ads_eV=-0.55,
        method="PBE-D3, slab",
        ref="Desai & Neurock 2003",
        doi=None,
        tier="discovery",
        facet="fcc(111)",
        slab_layers=4,
        supercell="3x3 tutorial slab; verify against source before strict parity",
        coverage="1 adsorbate per 3x3 surface cell in tutorial",
        frozen_layers="bottom 2 layers fixed in tutorial",
        functional="PBE family",
        dispersion="D3 in tutorial; reference dispersion treatment must be verified",
        source_type="primary literature context",
        source_url=None,
        reference_scope="context",
        notes="Use as mode/site context until exact low-coverage DFT value is verified.",
    ),
    ("Pd(111)", "NH3"): AdsorbMLReference(
        host="Pd(111)",
        adsorbate="NH3",
        binding_site="top (N-down)",
        E_ads_eV=None,
        method="DFT-GGA, periodic slab, 1/4 ML",
        ref="Herron, Tonelli & Mavrikakis 2012 Surface Science",
        doi="10.1016/j.susc.2012.07.003",
        tier="discovery",
        facet="fcc(111)",
        slab_layers=4,
        supercell="2x2 source slab; 3x3 tutorial slab",
        coverage="source 1/4 ML; tutorial 1 adsorbate per 3x3 surface cell",
        frozen_layers="top two layers relaxed in source; bottom 2 fixed in tutorial",
        functional="DFT-GGA",
        dispersion="D3 in tutorial; source value predates routine D3 correction",
        source_type="primary literature context",
        source_url="https://doi.org/10.1016/j.susc.2012.07.003",
        reference_scope="context",
        notes="Relevant ammonia-decomposition/small-molecule adsorption database case. Exact binding energy should be pulled from the source table before using an energy reference line.",
    ),
    ("Al2O3(0001)", "CO"): AdsorbMLReference(
        host="Al2O3(0001)",
        adsorbate="CO",
        binding_site="Al-top (C-down)",
        E_ads_eV=-0.35,
        method="PBE-D3, periodic slab",
        ref="Manassidis & Gillan 1994; OC22",
        doi="10.1021/acscatal.2c05426",
        tier="validation",
        facet="alpha-Al2O3(0001)",
        slab_layers=None,
        supercell="tutorial slab; exact OC22 slab id required before strict parity",
        coverage="1 adsorbate per tutorial surface cell",
        frozen_layers="bottom half fixed in tutorial",
        functional="PBE family",
        dispersion="D3 in tutorial; OC22 reference convention must be verified",
        source_type="OC22 + primary literature context",
        source_url="https://doi.org/10.1021/acscatal.2c05426",
        reference_scope="context",
        notes="Al2O3 termination/stoichiometry must match exactly before strict energy comparison.",
    ),
    ("Al2O3(0001)", "H2O"): AdsorbMLReference(
        host="Al2O3(0001)",
        adsorbate="H2O",
        binding_site="Al-top (O-down)",
        E_ads_eV=-0.85,
        method="PBE-D3, periodic slab",
        ref="Lodziana et al. 2004; OC22",
        doi="10.1021/acscatal.2c05426",
        tier="validation",
        facet="alpha-Al2O3(0001)",
        slab_layers=None,
        supercell="tutorial slab; exact OC22 slab id required before strict parity",
        coverage="1 adsorbate per tutorial surface cell",
        frozen_layers="bottom half fixed in tutorial",
        functional="PBE family",
        dispersion="D3 in tutorial; OC22/reference convention must be verified",
        source_type="OC22 + primary literature context",
        source_url="https://doi.org/10.1021/acscatal.2c05426",
        reference_scope="context",
        notes="Hydroxylation/dissociation and termination effects are major confounders; strict row must be molecular adsorption on matching termination.",
    ),
    ("Al2O3(0001)", "CH3OH"): AdsorbMLReference(
        host="Al2O3(0001)",
        adsorbate="CH3OH",
        binding_site="Al-top (O-down, H-bond to surface O)",
        E_ads_eV=-0.70,
        method="PBE-D3, periodic slab",
        ref="Tran et al. 2023 OC22",
        doi="10.1021/acscatal.2c05426",
        tier="discrepancy",  # subtle H-bonding; sparse oxide training in MPtrj
        facet="alpha-Al2O3(0001)",
        slab_layers=None,
        supercell="tutorial slab; exact OC22 slab id required before strict parity",
        coverage="1 adsorbate per tutorial surface cell",
        frozen_layers="bottom half fixed in tutorial",
        functional="PBE family",
        dispersion="D3 in tutorial; OC22/reference convention must be verified",
        source_type="OC22 context",
        source_url="https://doi.org/10.1021/acscatal.2c05426",
        reference_scope="context",
        notes="Candidate discrepancy case; do not over-interpret until exact OC22 matching datum is pinned.",
    ),
    ("Al2O3(0001)", "NH3"): AdsorbMLReference(
        host="Al2O3(0001)",
        adsorbate="NH3",
        binding_site="Al-top (N-down)",
        E_ads_eV=None,
        method="oxide-support / Lewis-acid-site literature context",
        ref="Kelber 2007 Surface Science Reports; McBriarty & Ellis 2016 Surface Science",
        doi="10.1016/j.surfrep.2006.12.003",
        tier="discrepancy",
        facet="alpha-Al2O3(0001)",
        slab_layers=None,
        supercell="tutorial slab; exact slab/termination reference required before strict parity",
        coverage="1 adsorbate per tutorial surface cell",
        frozen_layers="bottom half fixed in tutorial",
        functional="PBE family",
        dispersion="D3 in tutorial; reference convention must be selected",
        source_type="oxide-support chemistry context",
        source_url="https://doi.org/10.1016/j.surfrep.2006.12.003",
        reference_scope="context",
        notes="NH3 is relevant as a Lewis-base probe and NO-SCR feed molecule, but clean alpha-Al2O3(0001) molecular adsorption must be separated from hydroxylated/support-oxide chemistry before strict comparison.",
    ),
}


def get_adsorbml_reference(host: str, adsorbate: str) -> AdsorbMLReference | None:
    """Look up the AdsorbML panel reference for (host, adsorbate)."""
    return ADSORBML_REFERENCES.get((host, adsorbate))


ACTIVE_ADSORBML_PAIRS: tuple[tuple[str, str], ...] = (
    ("Cu(111)", "CO"),
    ("Cu(111)", "H2O"),
    ("Cu(111)", "CH3OH"),
    ("Pd(111)", "CO"),
    ("Pd(111)", "H2O"),
    ("Pd(111)", "CH3OH"),
    ("Al2O3(0001)", "CO"),
    ("Al2O3(0001)", "H2O"),
    ("Al2O3(0001)", "CH3OH"),
)

OPTIONAL_ADSORBML_CONTEXT_PAIRS: tuple[tuple[str, str], ...] = (
    ("Cu(111)", "NH3"),
    ("Pd(111)", "NH3"),
    ("Al2O3(0001)", "NH3"),
)


def active_adsorbml_references() -> dict[tuple[str, str], AdsorbMLReference]:
    """Return references for the active 9-pair notebook panel."""
    return {key: ADSORBML_REFERENCES[key] for key in ACTIVE_ADSORBML_PAIRS}


def optional_adsorbml_context_references() -> dict[tuple[str, str], AdsorbMLReference]:
    """Return optional phenomenon/context rows outside the active notebook panel."""
    return {key: ADSORBML_REFERENCES[key] for key in OPTIONAL_ADSORBML_CONTEXT_PAIRS}


def strict_adsorbml_references() -> dict[tuple[str, str], AdsorbMLReference]:
    """Return rows approved for strict apples-to-apples parity plotting."""
    return {
        key: ref for key, ref in ADSORBML_REFERENCES.items() if ref.strict_for_parity
    }


# Model-level OC157 molecule-surface relative-energy MAD values from the
# arXiv v3 foundation-model supplement. These are literature orientation
# scales only. The active tutorial calculations use MACE-MPA-0 with D3 disabled
# unless the workflow is explicitly reconfigured.
MACE_MPA0_OC157_MAD_EV = 0.28
MACE_MP0B3_OC157_MAD_EV = 0.38
LITERATURE_OC157_MAD_GUIDE_EV = MACE_MP0B3_OC157_MAD_EV
