"""Test structure generation: NaCl supercell."""

import pytest
from pymatgen.core import Lattice, Structure

from helpers.visualization import structure_summary_table


class TestNaClStructure:
    def test_unit_cell_atoms(self):
        nacl = Structure.from_spacegroup(
            "Fm-3m",
            Lattice.cubic(5.64),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        assert len(nacl) == 8  # 4 Na + 4 Cl in conventional cell

    def test_supercell_2x2x2(self):
        nacl = Structure.from_spacegroup(
            "Fm-3m",
            Lattice.cubic(5.64),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        nacl.make_supercell((2, 2, 2))
        assert len(nacl) == 64

    def test_supercell_3x3x3(self):
        nacl = Structure.from_spacegroup(
            "Fm-3m",
            Lattice.cubic(5.64),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        nacl.make_supercell((3, 3, 3))
        assert len(nacl) == 216

    def test_lattice_parameter(self):
        nacl = Structure.from_spacegroup(
            "Fm-3m",
            Lattice.cubic(5.64),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        assert nacl.lattice.a == pytest.approx(5.64)
        assert nacl.lattice.b == pytest.approx(5.64)
        assert nacl.lattice.c == pytest.approx(5.64)

    def test_ase_conversion(self, nacl_ase):
        assert len(nacl_ase) == 64
        symbols = set(nacl_ase.get_chemical_symbols())
        assert symbols == {"Na", "Cl"}
        assert nacl_ase.get_volume() == pytest.approx(5.64**3 * 8, rel=1e-3)

    def test_md_atomic_data_conversion(self, nacl_md_atoms):
        assert len(nacl_md_atoms.numbers) == 64
        assert len(nacl_md_atoms.coord) == 192  # 64 * 3
        assert nacl_md_atoms.pbc == [True, True, True]
        assert nacl_md_atoms.cell is not None

    def test_structure_summary_table(self, nacl_ase):
        df = structure_summary_table(nacl_ase)
        assert len(df) == 1
        assert df["Atoms"].iloc[0] == 64
        assert df["Density (g/cm^3)"].iloc[0] > 1.5
        assert df["Density (g/cm^3)"].iloc[0] < 3.0
