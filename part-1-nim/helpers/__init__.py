"""ALCHEMI NIM water-sorbent tutorial — helper modules."""

from .api_client import (
    async_run_bgr,
    async_run_bgr_or_load_cache,
    check_endpoint,
    run_bgr,
    run_bgr_or_load_cache,
)
from .cache import cache_exists, load_cache, save_cache
from .constants import (
    AMU_TO_G,
    ANGSTROM3_TO_CM3,
    BOLTZ_EV_K,
    EV_TO_KJ_MOL,
    KCAL_MOL_TO_EV,
    KE_CONV,
    KJ_MOL_TO_EV,
    P_CONV,
)
from .models import (
    BGRAtomicData,
    BGRReply,
    BGRRequest,
    OptimizationResult,
    ase_to_atomic_data,
    atomic_data_to_ase,
)
from .references import (
    ADSORBML_REFERENCES,
    MACE_MP0B3_OC157_MAD_EV,
    MACE_MPA0_OC157_MAD_EV,
    REFERENCES,
    S24_SUBCATEGORY_MAD_MEV,
    SMALL_MOLECULE_REFERENCES,
    AdsorbMLReference,
    AdsorptionReference,
    SmallMoleculeReference,
    get_adsorbml_reference,
    get_mad_meV,
    get_reference,
    get_small_molecule_reference,
)
from .oxide_slabs import (
    build_alpha_alumina_0001_slab,
    build_alpha_alumina_bulk,
    build_monoclinic_zro2_bulk,
    build_rutile_tio2_bulk,
    build_tio2_110_slab,
    build_zro2_m111_slab,
)
from .metal_slabs import (
    LATTICE_A_CU,
    LATTICE_A_PD,
    build_cu111_slab,
    build_cu_bulk,
    build_pd111_slab,
    build_pd_bulk,
)
from .config_search import (
    ADSORBATE_ORIENTATIONS,
    ADSORBATE_REGISTRY,
    Configuration,
    build_co,
    build_config_grid,
    build_h2o,
    build_methanol,
    find_al2o3_0001_sites,
    find_fcc_sites,
    sites_for_host,
)
from .zeolites import (
    build_h_cha,
    build_h_mfi,
    build_h_sapo34,
    build_siliceous_cha,
    build_siliceous_mfi,
)
from .throughput import (
    measure_batch_throughput,
    plot_throughput,
    sweep_batch_throughput,
)
from .surfaces import (
    build_adsorbate,
    build_slab,
    classify_relaxation,
    compute_adsorption_energy,
    compute_surface_displacement,
    find_central_site,
    make_active_mask,
    place_adsorbate,
)
from .visualization import (
    create_interactive_view,
    display_inline,
    display_widgets_row,
    render_structure_ovito,
    structure_summary_table,
)

__all__ = [
    # api_client
    "check_endpoint",
    "run_bgr",
    "run_bgr_or_load_cache",
    "async_run_bgr",
    "async_run_bgr_or_load_cache",
    # cache
    "save_cache",
    "load_cache",
    "cache_exists",
    # constants
    "KE_CONV",
    "BOLTZ_EV_K",
    "P_CONV",
    "AMU_TO_G",
    "ANGSTROM3_TO_CM3",
    "KCAL_MOL_TO_EV",
    "KJ_MOL_TO_EV",
    "EV_TO_KJ_MOL",
    # models
    "BGRAtomicData",
    "BGRRequest",
    "BGRReply",
    "OptimizationResult",
    "ase_to_atomic_data",
    "atomic_data_to_ase",
    # surfaces
    "build_slab",
    "make_active_mask",
    "find_central_site",
    "build_adsorbate",
    "place_adsorbate",
    "classify_relaxation",
    "compute_adsorption_energy",
    "compute_surface_displacement",
    # visualization
    "render_structure_ovito",
    "create_interactive_view",
    "display_widgets_row",
    "display_inline",
    "structure_summary_table",
    # oxide_slabs
    "build_alpha_alumina_bulk",
    "build_rutile_tio2_bulk",
    "build_monoclinic_zro2_bulk",
    "build_alpha_alumina_0001_slab",
    "build_tio2_110_slab",
    "build_zro2_m111_slab",
    # metal_slabs
    "LATTICE_A_CU",
    "LATTICE_A_PD",
    "build_cu_bulk",
    "build_pd_bulk",
    "build_cu111_slab",
    "build_pd111_slab",
    # config_search
    "Configuration",
    "ADSORBATE_REGISTRY",
    "ADSORBATE_ORIENTATIONS",
    "build_co",
    "build_h2o",
    "build_methanol",
    "find_fcc_sites",
    "find_al2o3_0001_sites",
    "sites_for_host",
    "build_config_grid",
    # zeolites
    "build_siliceous_cha",
    "build_siliceous_mfi",
    "build_h_cha",
    "build_h_mfi",
    "build_h_sapo34",
    # throughput
    "measure_batch_throughput",
    "sweep_batch_throughput",
    "plot_throughput",
    # references
    "AdsorptionReference",
    "SmallMoleculeReference",
    "AdsorbMLReference",
    "REFERENCES",
    "S24_SUBCATEGORY_MAD_MEV",
    "SMALL_MOLECULE_REFERENCES",
    "ADSORBML_REFERENCES",
    "MACE_MPA0_OC157_MAD_EV",
    "MACE_MP0B3_OC157_MAD_EV",
    "get_reference",
    "get_mad_meV",
    "get_small_molecule_reference",
    "get_adsorbml_reference",
]
