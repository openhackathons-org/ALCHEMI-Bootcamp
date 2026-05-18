"""Test that all required packages and helper modules import correctly."""

import pytest


class TestPackageImports:
    def test_ase(self):
        import ase  # noqa: F401

    def test_pymatgen(self):
        from pymatgen.core import Lattice, Structure  # noqa: F401

    def test_pydantic(self):
        import pydantic  # noqa: F401

    def test_numpy(self):
        import numpy  # noqa: F401

    def test_matplotlib(self):
        import matplotlib  # noqa: F401

    def test_pandas(self):
        import pandas  # noqa: F401

    def test_ovito(self):
        try:
            import ovito  # noqa: F401
        except ImportError as exc:
            if "libOpenGL.so.0" in str(exc):
                pytest.skip("OVITO is installed, but this environment is missing libOpenGL.so.0")
            raise


class TestHelperImports:
    def test_models_classes(self):
        from helpers.models import (  # noqa: F401
            AtomicStructurePayload,
            OptimizationResult,
            RelaxationBatchResult,
            RelaxationRequest,
        )

    def test_models_functions(self):
        from helpers.models import (  # noqa: F401
            ase_to_atomic_data,
            atomic_data_to_ase,
        )

    def test_cache(self):
        from helpers.cache import cache_exists, load_cache, save_cache

        assert callable(cache_exists)
        assert callable(load_cache)
        assert callable(save_cache)

    def test_relaxation_backend_exports(self):
        from helpers.relaxation_backends import (  # noqa: F401
            BackendUnavailableError,
            RelaxationBackendConfig,
            ToolkitD3BJConfig,
            check_toolkit_native_api,
            get_relaxation_backend,
        )

    def test_visualization(self):
        from helpers.visualization import (  # noqa: F401
            display_inline,
            render_structure_ovito,
            structure_summary_table,
        )

    def test_surfaces(self):
        from helpers.surfaces import (  # noqa: F401
            build_slab,
            build_adsorbate,
            place_adsorbate,
            find_central_site,
            compute_adsorption_energy,
        )

    def test_package_init_exports(self):
        from helpers import (  # noqa: F401
            ADSORPTION_ENERGY_FORMULA,
            AtomicStructurePayload,
            compute_adsorption_energy_ev,
            get_relaxation_backend,
            structure_summary_table,
            build_slab,
            place_adsorbate,
            surface_screen_expected_counts,
            KJ_MOL_TO_EV,
        )
