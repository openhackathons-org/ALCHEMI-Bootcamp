"""Test visualization utilities."""

import os
import tempfile

import pytest
import pandas as pd
from ase import Atoms
from ase.io import write as ase_write

from helpers.visualization import (
    _clean_atoms_for_ovito,
    _trajectory_with_whole_adsorbate,
    render_structure_ovito,
    structure_summary_table,
    subscript_formula_html,
    subscript_formula_markdown,
)


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


class TestFormulaFormatting:
    def test_html_subscripts_common_formulas(self):
        label = subscript_formula_html("H2O on TiO2(110) with CH3OH")
        assert label == "H<sub>2</sub>O on TiO<sub>2</sub>(110) with CH<sub>3</sub>OH"

    def test_markdown_subscripts_skip_code_spans(self):
        text = subscript_formula_markdown("H2O and `H2O` on Al2O3")
        assert text == "H<sub>2</sub>O and `H2O` on Al<sub>2</sub>O<sub>3</sub>"


class TestOVITOCellDisplayPreparation:
    def test_wraps_display_copy_into_periodic_cell(self):
        atoms = Atoms(
            "TiOH2",
            positions=[
                [0.0, -13.0, 8.0],
                [1.0, -6.5, 8.5],
                [2.0, 14.0, 9.0],
                [2.5, 14.5, 9.5],
            ],
            cell=[5.0, 13.0, 25.0],
            pbc=True,
        )
        prepared = _clean_atoms_for_ovito(atoms, wrap_periodic_cell=True)

        scaled = prepared.get_scaled_positions(wrap=False)
        assert scaled.min() >= -1e-7
        assert scaled.max() < 1.0 + 1e-7
        assert atoms.positions[:, 1].min() < 0.0


class TestTrajectoryPreparation:
    def test_whole_adsorbate_trajectory_path_does_not_raise(self, tmp_path):
        frames = [
            Atoms(
                "TiNH3",
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.2, 0.2, 1.0],
                    [0.3, 0.2, 1.8],
                    [0.2, 0.3, 1.9],
                    [0.1, 0.2, 1.9],
                ],
                cell=[6.0, 6.0, 10.0],
                pbc=True,
            )
        ]
        source = tmp_path / "input.extxyz"
        output = tmp_path / "whole.extxyz"
        ase_write(source, frames, format="extxyz")

        result = _trajectory_with_whole_adsorbate(source, output)

        assert result in {source, output}
        assert result.exists()


class TestSurfaceScreenWinnerArtifacts:
    def test_live_ranking_table_can_use_existing_relaxed_structures_without_csv(
        self, tmp_path
    ):
        from helpers.surface_screen_widgets import _load_ranked_winners

        root = tmp_path / "surface_screen"
        (root / "tables").mkdir(parents=True)
        (root / "structures" / "relaxed_adsorption").mkdir(parents=True)
        ranking_df = pd.DataFrame(
            [
                {
                    "adsorbate": "H2O",
                    "host": "TiO2(110)",
                    "rank_within_adsorbate": 1,
                    "best_E_ads_eV": -0.42,
                    "best_label": "H2O_TiO2_110_top",
                    "best_final_site": "top",
                    "best_optimizer_nsteps": 12,
                }
            ]
        )

        ranked, pair_summary_path = _load_ranked_winners(
            surface_pair_summary_df=ranking_df,
            surface_screen_paths={"root": root},
        )

        assert pair_summary_path == root / "tables" / "pair_summary.csv"
        assert ranked["best_label"].tolist() == ["H2O_TiO2_110_top"]


class TestOVITORender:
    def test_render_creates_file(self, nacl_ase):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_render.png")
            try:
                result = render_structure_ovito(nacl_ase, output_path=out_path)
            except ImportError as exc:
                if "libOpenGL.so.0" in str(exc) or "DataCollection" in str(exc):
                    pytest.skip(
                        f"OVITO rendering is unavailable in this environment: {exc}"
                    )
                raise
            assert os.path.isfile(result)
            assert os.path.getsize(result) > 0

    def test_render_output_exists(self, output_dir):
        render_path = os.path.join(output_dir, "nacl_render.png")
        if not os.path.isfile(render_path):
            pytest.skip("OVITO render not yet generated")
        assert os.path.getsize(render_path) > 1000  # should be a real PNG
