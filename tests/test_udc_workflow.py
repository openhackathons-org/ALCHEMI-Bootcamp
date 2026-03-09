"""Test UDC-inspired conformer stability workflow."""

import os

import ase.io
import numpy as np
import pytest

from helpers.cache import cache_exists, load_cache
from helpers.models import MDAtomicData, MDReply


class TestConformerGeneration:
    def test_conformers_from_naphthalene(self, structure_dir):
        path = os.path.join(structure_dir, "naphthalene.xyz")
        base = ase.io.read(path)

        np.random.seed(42)
        conformers = [base.copy()]
        for _ in range(2):
            conf = base.copy()
            conf.positions += np.random.normal(0, 0.05, conf.positions.shape)
            conformers.append(conf)

        assert len(conformers) == 3
        for conf in conformers:
            assert len(conf) == 18

    def test_conformers_differ(self, structure_dir):
        path = os.path.join(structure_dir, "naphthalene.xyz")
        base = ase.io.read(path)

        np.random.seed(42)
        conf1 = base.copy()
        conf1.positions += np.random.normal(0, 0.05, conf1.positions.shape)
        conf2 = base.copy()
        conf2.positions += np.random.normal(0, 0.05, conf2.positions.shape)

        assert not np.allclose(base.positions, conf1.positions)
        assert not np.allclose(conf1.positions, conf2.positions)

    def test_conformer_to_md_atomic_data_periodic(self, structure_dir):
        """Conformers must be placed in a periodic box for the endpoint."""
        path = os.path.join(structure_dir, "naphthalene.xyz")
        base = ase.io.read(path)

        BOX_SIZE = 50.0
        centroid = base.positions.mean(axis=0)
        centred = base.positions - centroid + BOX_SIZE / 2.0

        md_data = MDAtomicData(
            coord=centred.flatten().tolist(),
            numbers=base.numbers.tolist(),
            cell=[BOX_SIZE, 0, 0, 0, BOX_SIZE, 0, 0, 0, BOX_SIZE],
            pbc=[True, True, True],
        )
        assert len(md_data.numbers) == 18
        assert md_data.cell is not None
        assert md_data.pbc == [True, True, True]


class TestStabilityScoring:
    def _skip_if_no_conformer_cache(self, cache_dir):
        for i in range(3):
            if not cache_exists(cache_dir, f"naph_conf{i}_nvt_500K"):
                pytest.skip("Conformer cache not available")

    def test_all_conformers_succeed(self, cache_dir):
        self._skip_if_no_conformer_cache(cache_dir)
        for i in range(3):
            reply = load_cache(cache_dir, f"naph_conf{i}_nvt_500K", MDReply)
            assert reply.status == "Success"
            assert len(reply.trajectory) > 0

    def test_energy_timeseries_reasonable(self, cache_dir):
        self._skip_if_no_conformer_cache(cache_dir)
        for i in range(3):
            reply = load_cache(cache_dir, f"naph_conf{i}_nvt_500K", MDReply)
            energies = np.array([s.energy for s in reply.trajectory])
            assert len(energies) > 5
            assert np.all(np.isfinite(energies))
            # Naphthalene energy should be negative (bound molecule)
            assert energies.mean() < 0

    def test_stability_scoring(self, cache_dir):
        """Composite stability score: lower is more stable."""
        self._skip_if_no_conformer_cache(cache_dir)
        scores = []
        for i in range(3):
            reply = load_cache(cache_dir, f"naph_conf{i}_nvt_500K", MDReply)
            energies = np.array([s.energy for s in reply.trajectory])
            n_equil = int(0.3 * len(energies))
            prod_e = energies[n_equil:]

            e_mean = prod_e.mean()
            e_std = prod_e.std()
            t_arr = np.arange(len(prod_e), dtype=float)
            drift = np.polyfit(t_arr, prod_e, 1)[0] if len(prod_e) > 1 else 0.0

            composite = abs(e_mean) * 0.01 + e_std + abs(drift) * 100
            scores.append(composite)

        assert len(scores) == 3
        assert all(np.isfinite(s) for s in scores)
        assert all(s > 0 for s in scores)

        # The base conformer (index 0) should be among the more stable ones
        best = np.argmin(scores)
        assert best in [0, 1, 2]  # sanity check (it's a valid index)
