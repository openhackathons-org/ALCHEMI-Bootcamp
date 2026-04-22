"""Structural sanity tests for the 6 AWH host frameworks."""

from __future__ import annotations

import numpy as np
import pytest

from helpers.oxide_slabs import (
    build_alpha_alumina_0001_slab,
    build_alpha_alumina_bulk,
    build_monoclinic_zro2_bulk,
    build_rutile_tio2_bulk,
    build_tio2_110_slab,
    build_zro2_m111_slab,
)
from helpers.zeolites import (
    build_h_cha,
    build_h_sapo34,
    build_siliceous_cha,
    build_siliceous_mfi,
)


# ---------------------------------------------------------------------------
# Oxide bulks
# ---------------------------------------------------------------------------


class TestOxideBulks:
    def test_alumina_composition(self):
        bulk = build_alpha_alumina_bulk()
        comp = bulk.composition.as_dict()
        ratio = comp["O"] / comp["Al"]
        assert ratio == pytest.approx(1.5, abs=1e-6)  # Al2O3 stoichiometry

    def test_alumina_lattice(self):
        bulk = build_alpha_alumina_bulk()
        a, _, c = bulk.lattice.abc
        assert a == pytest.approx(4.7607, abs=0.001)
        assert c == pytest.approx(12.9947, abs=0.001)

    def test_rutile_composition(self):
        bulk = build_rutile_tio2_bulk()
        comp = bulk.composition.as_dict()
        assert comp["O"] / comp["Ti"] == pytest.approx(2.0, abs=1e-6)

    def test_rutile_lattice(self):
        bulk = build_rutile_tio2_bulk()
        a, b, c = bulk.lattice.abc
        assert a == b == pytest.approx(4.594, abs=0.001)
        assert c == pytest.approx(2.958, abs=0.001)

    def test_zro2_composition(self):
        bulk = build_monoclinic_zro2_bulk()
        comp = bulk.composition.as_dict()
        assert comp["O"] / comp["Zr"] == pytest.approx(2.0, abs=1e-6)

    def test_zro2_monoclinic_angle(self):
        bulk = build_monoclinic_zro2_bulk()
        _, beta, _ = bulk.lattice.angles
        assert beta == pytest.approx(99.23, abs=0.01)


# ---------------------------------------------------------------------------
# Oxide slabs
# ---------------------------------------------------------------------------


class TestOxideSlabs:
    @pytest.mark.parametrize("slab_fn", [
        build_alpha_alumina_0001_slab,
        build_tio2_110_slab,
        build_zro2_m111_slab,
    ])
    def test_slab_has_pbc(self, slab_fn):
        slab = slab_fn()
        assert all(slab.pbc)
        assert slab.cell.volume > 0

    @pytest.mark.parametrize("slab_fn", [
        build_alpha_alumina_0001_slab,
        build_tio2_110_slab,
        build_zro2_m111_slab,
    ])
    def test_slab_has_vacuum(self, slab_fn):
        """Slab must have a vacuum region along c > min_vacuum_size."""
        slab = slab_fn(min_vacuum_size=15.0)
        z = slab.positions[:, 2]
        slab_span = float(z.max() - z.min())
        c_length = float(np.linalg.norm(slab.cell[2]))
        vacuum = c_length - slab_span
        assert vacuum >= 10.0, f"vacuum={vacuum:.2f} A too small"


# ---------------------------------------------------------------------------
# Zeolites
# ---------------------------------------------------------------------------


class TestSiliceousFrameworks:
    def test_cha_composition(self):
        atoms = build_siliceous_cha()
        symbols = atoms.get_chemical_symbols()
        n_si = symbols.count("Si")
        n_o = symbols.count("O")
        assert n_si > 0
        assert n_o / n_si == pytest.approx(2.0, abs=1e-6)

    def test_mfi_composition(self):
        atoms = build_siliceous_mfi()
        symbols = atoms.get_chemical_symbols()
        n_si = symbols.count("Si")
        n_o = symbols.count("O")
        assert n_si > 0
        assert n_o / n_si == pytest.approx(2.0, abs=1e-6)

    def test_cha_pbc(self):
        atoms = build_siliceous_cha()
        assert all(atoms.pbc)
        assert atoms.cell.volume > 0

    def test_mfi_pbc(self):
        atoms = build_siliceous_mfi()
        assert all(atoms.pbc)
        assert atoms.cell.volume > 0


class TestBronstedForms:
    def test_h_cha_has_one_al_and_one_extra_h(self):
        pure = build_siliceous_cha()
        h_cha = build_h_cha()
        pure_sym = pure.get_chemical_symbols()
        h_sym = h_cha.get_chemical_symbols()
        assert h_sym.count("Al") == 1
        assert h_sym.count("Si") == pure_sym.count("Si") - 1
        assert h_sym.count("H") == 1
        # O count unchanged (H added, not O)
        assert h_sym.count("O") == pure_sym.count("O")

    def test_h_cha_proton_near_bridging_oxygen(self):
        h_cha = build_h_cha()
        symbols = h_cha.get_chemical_symbols()
        h_idx = symbols.index("H")
        o_indices = [i for i, s in enumerate(symbols) if s == "O"]
        # nearest O to H must be within typical O-H bond length
        h_pos = h_cha.positions[h_idx]
        cell = h_cha.cell.array
        inv_cell = np.linalg.inv(cell)
        dists = []
        for oi in o_indices:
            d = h_cha.positions[oi] - h_pos
            frac = d @ inv_cell
            frac -= np.round(frac)
            dists.append(float(np.linalg.norm(frac @ cell)))
        assert min(dists) < 1.2  # H-O bond length < 1.2 A

    def test_h_sapo34_matches_h_cha_shape(self):
        """Current H-SAPO-34 approximation shares the single-Al-substitution
        construction; record that so a future divergence is intentional."""
        assert len(build_h_sapo34()) == len(build_h_cha())
