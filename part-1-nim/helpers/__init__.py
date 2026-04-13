"""ALCHEMI OER Catalyst Screening — helper modules."""

from .api_client import (
    async_run_bgr,
    async_run_bgr_or_load_cache,
    check_endpoint,
    run_bgr,
    run_bgr_or_load_cache,
)
from .cache import cache_exists, load_cache, save_cache
from .constants import EV_PER_OER_STEP
from .models import (
    BGRAtomicData,
    BGRReply,
    BGRRequest,
    OptimizationResult,
    ase_to_atomic_data,
    atomic_data_to_ase,
)
from .surfaces import (
    build_adsorbate,
    build_rutile_bulk,
    build_slab,
    classify_relaxation,
    compute_adsorption_energy,
    compute_surface_displacement,
    find_bridge_site,
    find_central_site,
    find_cus_sites,
    find_site,
    make_active_mask,
    place_adsorbate,
)
from .visualization import (
    create_interactive_view,
    display_inline,
    display_widgets_row,
    plot_electrolysis_diagram,
    plot_oer_energy_ladders,
    render_structure_ovito,
    structure_summary_table,
)

__all__ = [
    "BGRAtomicData",
    "BGRRequest",
    "OptimizationResult",
    "BGRReply",
    "EV_PER_OER_STEP",
    "ase_to_atomic_data",
    "atomic_data_to_ase",
    "save_cache",
    "load_cache",
    "cache_exists",
    "check_endpoint",
    "run_bgr",
    "run_bgr_or_load_cache",
    "async_run_bgr",
    "async_run_bgr_or_load_cache",
    "render_structure_ovito",
    "create_interactive_view",
    "display_widgets_row",
    "display_inline",
    "plot_electrolysis_diagram",
    "plot_oer_energy_ladders",
    "structure_summary_table",
    "build_rutile_bulk",
    "build_slab",
    "make_active_mask",
    "find_central_site",
    "find_cus_sites",
    "find_bridge_site",
    "find_site",
    "build_adsorbate",
    "place_adsorbate",
    "classify_relaxation",
    "compute_adsorption_energy",
    "compute_surface_displacement",
]
