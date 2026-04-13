"""Test visualization utilities."""

import os
import tempfile

import pytest

from helpers.visualization import render_structure_ovito, structure_summary_table


class TestStructureSummary:
    def test_summary_columns(self, nacl_ase):
        df = structure_summary_table(nacl_ase)
        expected_cols = {
            "Formula",
            "Atoms",
            "a (A)",
            "b (A)",
            "c (A)",
            "Volume (A^3)",
            "Density (g/cm^3)",
        }
        assert set(df.columns) == expected_cols

    def test_summary_values(self, nacl_ase):
        df = structure_summary_table(nacl_ase)
        assert df["Atoms"].iloc[0] == 64
        assert df["a (A)"].iloc[0] == pytest.approx(5.64 * 2, abs=0.1)
        assert df["Volume (A^3)"].iloc[0] > 0
        assert df["Density (g/cm^3)"].iloc[0] > 1.0


class TestOVITORender:
    def test_render_creates_file(self, nacl_ase):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_render.png")
            result = render_structure_ovito(nacl_ase, output_path=out_path)
            assert os.path.isfile(result)
            assert os.path.getsize(result) > 0

    def test_render_output_exists(self, output_dir):
        render_path = os.path.join(output_dir, "nacl_render.png")
        if not os.path.isfile(render_path):
            pytest.skip("OVITO render not yet generated")
        assert os.path.getsize(render_path) > 1000  # should be a real PNG


