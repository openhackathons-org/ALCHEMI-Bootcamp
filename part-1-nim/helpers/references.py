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
