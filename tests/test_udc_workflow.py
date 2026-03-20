"""Tests for the OLED material screening workflow."""

import numpy as np
import pytest
from rdkit import Chem

from helpers.analysis import check_bond_integrity, compute_rmsd, kabsch_rmsd
from helpers.cache import cache_exists, load_cache
from helpers.conformers import (
    compute_n_conformers,
    deduplicate_conformers,
    filter_by_energy,
    generate_conformers,
)
from helpers.models import BMDAtomicData, BMDReply


# ---------------------------------------------------------------------------
# OLED candidate SMILES (same as notebook)
# ---------------------------------------------------------------------------
OLED_SMILES = {
    "CBP": "c1ccc(-c2ccc(-n3c4ccccc4c4ccccc43)cc2)cc1",
    "NPB": "c1ccc(-c2ccc(-N(c3ccccc3)c3ccc4ccccc4c3)cc2)cc1",
    "mCP": "c1ccc2c(c1)c1ccccc1n2-c1cccc(-n2c3ccccc3c3ccccc32)c1",
    "BCP": "Cc1ccc2c(-c3ccccc3)c3ccc4cc(-c5ccccc5)c(C)nc4c3nc2c1",
    "TPBi": "c1ccc(-c2nc3ccccc3[nH]2)cc1",
}


class TestSMILESValidation:
    @pytest.mark.parametrize("name,smi", list(OLED_SMILES.items()))
    def test_smiles_parseable(self, name, smi):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"{name} SMILES failed to parse"

    @pytest.mark.parametrize("name,smi", list(OLED_SMILES.items()))
    def test_smiles_has_atoms(self, name, smi):
        mol = Chem.MolFromSmiles(smi)
        assert mol.GetNumAtoms() > 0

    def test_cbp_atom_count(self):
        mol = Chem.AddHs(Chem.MolFromSmiles(OLED_SMILES["CBP"]))
        assert mol.GetNumAtoms() > 30  # CBP is a large molecule

    def test_bcp_atom_count(self):
        mol = Chem.AddHs(Chem.MolFromSmiles(OLED_SMILES["BCP"]))
        assert mol.GetNumAtoms() > 30  # BCP with methyls and phenyl groups
        # The exact H count depends on the SMILES; verify at least heavy atoms


class TestConformerGeneration:
    def test_compute_n_conformers_small(self):
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene — 0 rot bonds
        assert compute_n_conformers(mol) == 200

    def test_compute_n_conformers_flexible(self):
        mol = Chem.MolFromSmiles("CCCCCCCC")  # octane — many rot bonds
        n = compute_n_conformers(mol)
        assert 200 <= n <= 1000

    def test_generate_conformers_benzene(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = generate_conformers(mol, n_confs=10, seed=42)
        assert mol.GetNumConformers() >= 1

    def test_generate_conformers_cbp(self):
        mol = Chem.MolFromSmiles(OLED_SMILES["CBP"])
        mol = generate_conformers(mol, n_confs=5, seed=42)
        assert mol.GetNumConformers() >= 1

    def test_conformers_differ(self):
        mol = Chem.MolFromSmiles(OLED_SMILES["mCP"])
        mol = generate_conformers(mol, n_confs=10, seed=42)
        if mol.GetNumConformers() >= 2:
            c0 = mol.GetConformer(0).GetPositions()
            c1 = mol.GetConformer(1).GetPositions()
            assert not np.allclose(c0, c1)


class TestVacuumBox:
    def test_conformer_to_md_atomic_data_periodic(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        from rdkit.Chem import AllChem

        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        conf = mol.GetConformer()
        pos = np.array(conf.GetPositions())
        numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]

        BOX_SIZE = 50.0
        centroid = pos.mean(axis=0)
        centred = pos - centroid + BOX_SIZE / 2.0

        md_data = BMDAtomicData(
            coord=centred.flatten().tolist(),
            numbers=numbers,
            cell=[BOX_SIZE, 0, 0, 0, BOX_SIZE, 0, 0, 0, BOX_SIZE],
            pbc=[True, True, True],
        )
        assert md_data.cell is not None
        assert md_data.pbc == [True, True, True]
        assert len(md_data.numbers) == mol.GetNumAtoms()


class TestEnergyFiltering:
    def test_all_pass_within_threshold(self):
        energies = np.array([-10.0, -9.99, -9.98])
        mask = filter_by_energy(energies, threshold_kcal=3.0)
        assert mask.all()

    def test_high_energy_filtered(self):
        # 3 kcal/mol = 0.1302 eV; -10.0 to -9.9 = 0.1 eV (within), -5.0 = 5.0 eV (outside)
        energies = np.array([-10.0, -9.9, -5.0])
        mask = filter_by_energy(energies, threshold_kcal=3.0)
        assert mask[0] and mask[1] and not mask[2]

    def test_single_conformer(self):
        energies = np.array([-7.0])
        mask = filter_by_energy(energies, threshold_kcal=3.0)
        assert mask[0]


class TestDeduplication:
    def test_identical_coords_deduplicated(self):
        rng = np.random.default_rng(42)
        coords = rng.normal(size=(10, 3))
        # Third set uses genuinely different coordinates (not just a translation)
        coords_diff = rng.normal(size=(10, 3)) * 10
        coords_list = [coords, coords.copy(), coords_diff]
        energies = np.array([0.0, 0.01, 0.5])
        unique = deduplicate_conformers(coords_list, energies, rmsd_threshold=0.125)
        assert len(unique) == 2  # first two are duplicates

    def test_all_distinct(self):
        rng = np.random.default_rng(42)
        coords_list = [rng.normal(size=(10, 3)) * 10 for _ in range(3)]
        energies = np.array([0.0, 0.1, 0.2])
        unique = deduplicate_conformers(coords_list, energies, rmsd_threshold=0.125)
        assert len(unique) == 3


class TestKabschRMSD:
    def test_identical(self):
        coords = np.random.default_rng(42).normal(size=(20, 3))
        assert kabsch_rmsd(coords, coords) < 1e-10

    def test_translated(self):
        coords = np.random.default_rng(42).normal(size=(20, 3))
        assert kabsch_rmsd(coords, coords + 100.0) < 1e-10

    def test_rotated(self):
        coords = np.random.default_rng(42).normal(size=(20, 3))
        # 90-degree rotation about z-axis
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        rotated = coords @ R.T
        assert kabsch_rmsd(coords, rotated) < 1e-8

    def test_different_coords(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(20, 3))
        b = rng.normal(size=(20, 3)) * 5
        assert kabsch_rmsd(a, b) > 0.1


class TestComputeRMSD:
    def test_first_frame_zero(self):
        import ase

        atoms = ase.Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms.info["md_time"] = 0.0
        frames = [atoms.copy() for _ in range(5)]
        for i, f in enumerate(frames):
            f.info["md_time"] = float(i)
        times, rmsds = compute_rmsd(frames)
        assert rmsds[0] == pytest.approx(0.0)
        assert len(times) == 5

    def test_rmsd_increases_with_displacement(self):
        import ase

        frames = []
        for i in range(5):
            atoms = ase.Atoms("H2", positions=[[0, 0, 0], [1 + 0.1 * i, 0, 0]])
            atoms.info["md_time"] = float(i)
            frames.append(atoms)
        _, rmsds = compute_rmsd(frames)
        assert rmsds[-1] > rmsds[0]


class TestBondIntegrity:
    def test_intact_trajectory(self):
        import ase

        frames = []
        for _ in range(5):
            atoms = ase.Atoms(
                "CH4",
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.63, 0.63, 0.63],
                    [-0.63, -0.63, 0.63],
                    [-0.63, 0.63, -0.63],
                    [0.63, -0.63, -0.63],
                ],
            )
            frames.append(atoms)
        result = check_bond_integrity(frames)
        assert result["integrity_score"] == 1.0
        assert result["n_broken"] == 0
        assert result["n_formed"] == 0


class TestCompositeScoring:
    def test_scoring_formula(self):
        e_std = 0.1
        drift = 0.001
        rmsd_mean = 0.5
        bond_integrity = 1.0

        composite = (
            e_std * 10.0
            + abs(drift) * 1000.0
            + rmsd_mean * 2.0
            + (1.0 - bond_integrity) * 50.0
        )
        assert composite > 0
        assert np.isfinite(composite)

    def test_perfect_scores_beat_imperfect(self):
        good = 0.1 * 10 + 0.0001 * 1000 + 0.3 * 2 + 0.0 * 50
        bad = 0.5 * 10 + 0.01 * 1000 + 2.0 * 2 + 0.1 * 50
        assert good < bad


class TestCacheValidation:
    """Validate OLED cache files if they exist."""

    OLED_BGR_LABELS = [
        "oled_cbp_bgr_confs",
        "oled_npb_bgr_confs",
        "oled_mcp_bgr_confs",
        "oled_bcp_bgr_confs",
        "oled_tpbi_bgr_confs",
    ]
    OLED_MD_LABELS = [
        "oled_cbp_nvt_500K",
        "oled_npb_nvt_500K",
        "oled_mcp_nvt_500K",
        "oled_bcp_nvt_500K",
        "oled_tpbi_nvt_500K",
    ]

    @pytest.mark.parametrize("label", OLED_MD_LABELS)
    def test_md_cache_valid(self, conformer_cache_dir, label):
        if not cache_exists(conformer_cache_dir, label):
            pytest.skip(f"Cache {label} not available")
        reply = load_cache(conformer_cache_dir, label, BMDReply)
        assert reply.status == "Success"
        assert len(reply.trajectory) > 0
        energies = np.array([s.energy for s in reply.trajectory])
        assert np.all(np.isfinite(energies))
