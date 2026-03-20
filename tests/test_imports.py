"""Test that all required packages and helper modules import correctly."""

import pytest


class TestPackageImports:
    def test_ase(self):
        import ase  # noqa: F401

    def test_pymatgen(self):
        from pymatgen.core import Lattice, Structure  # noqa: F401

    def test_pydantic(self):
        import pydantic  # noqa: F401

    def test_requests(self):
        import requests  # noqa: F401

    def test_rdkit(self):
        import rdkit  # noqa: F401

    def test_numpy(self):
        import numpy  # noqa: F401

    def test_matplotlib(self):
        import matplotlib  # noqa: F401

    def test_pandas(self):
        import pandas  # noqa: F401

    def test_ovito(self):
        import ovito  # noqa: F401


class TestHelperImports:
    def test_models_classes(self):
        from helpers.models import (  # noqa: F401
            BMDAtomicData,
            BMDConfig,
            BMDReply,
            BMDRequest,
            BMDSnapshot,
            BGRAtomicData,
            BGRReply,
            BGRRequest,
            OptimizationResult,
        )

    def test_models_constants(self):
        from helpers.models import BOLTZ_EV_K, KE_CONV, P_CONV

        assert KE_CONV == pytest.approx(103.64269667160806)
        assert BOLTZ_EV_K == pytest.approx(8.617333262145179e-05)
        assert P_CONV == pytest.approx(1.602176634e6)

    def test_models_functions(self):
        from helpers.models import (  # noqa: F401
            ase_to_atomic_data,
            ase_to_md_atomic_data,
            atomic_data_to_ase,
            read_to_bmd_atomic_data,
        )

    def test_cache(self):
        from helpers.cache import cache_exists, load_cache, save_cache

        assert callable(cache_exists)
        assert callable(load_cache)
        assert callable(save_cache)

    def test_api_client(self):
        from helpers.api_client import (  # noqa: F401
            check_endpoint,
            run_bgr,
            run_bgr_or_load_cache,
            run_md,
            run_md_or_load_cache,
        )

    def test_analysis(self):
        from helpers.analysis import (  # noqa: F401
            compute_density,
            compute_msd,
            compute_rdf,
            estimate_diffusion_coefficient,
            extract_thermo_timeseries,
            pick_production_window,
            thermal_expansion_proxy,
            trajectory_to_ase_list,
        )

    def test_visualization(self):
        from helpers.visualization import (  # noqa: F401
            display_inline,
            render_structure_ovito,
            structure_summary_table,
        )

    def test_package_init_exports(self):
        from helpers import (  # noqa: F401
            BMDAtomicData,
            check_endpoint,
            compute_rdf,
            extract_thermo_timeseries,
            run_md_or_load_cache,
            structure_summary_table,
        )
