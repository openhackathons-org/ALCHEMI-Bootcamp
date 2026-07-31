# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ALCHEMI part 2 toolkit -- notebook helper modules.

Flat re-exports so notebooks can ``from helpers import ...`` everything
they need in one line.

Container-only sub-modules (analysis, dynamics, hooks, io) require torch
and/or nvalchemi. Their imports are wrapped in ``try/except ImportError``
so host-side scripts (running in the alchemi-playbook conda env, which
ships only numpy/ase/matplotlib) can still import the host-safe helpers
(constants, diffusion, visualization). Names from a sub-module that
fails to import simply won't be present on the package; consumers that
need them will hit a clean ``ImportError`` at use site.
"""

# --- Always available (numpy / pure Python) -----------------------------
from .constants import (
    AMU_OVER_A3_TO_G_CM3,
    D3_PRESETS,
    MAX_FORCE_CLAMP,
    P_1ATM,
    STAGE_COLORS,
    STAGE_LABELS,
    status_by_stage,
    warmup_stage_names,
)
from .diffusion import (
    compute_com_msd_numpy,
    fit_diffusion_coefficient,
)

# --- Container-only (torch / nvalchemi / pydantic) ----------------------
try:
    from .config import Config
except ImportError:
    pass

try:
    from .analysis import (
        _mol_inertia_eigvecs,
        compute_com_msd,
        compute_mol_axes,
        compute_molecule_axes,
        compute_msd,
        compute_rACF,
        compute_rdf,
        compute_rotational_acf,
        compute_S0_from_frames,
        compute_S0_tail,
        min_pbc_distance,
    )
except ImportError:
    pass

try:
    from .dynamics import (
        DYNAMICS_SCALARS,
        batch_to_ase,
        compute_density,
        density_scalar,
        pressure_scalar,
        volume_scalar,
    )
except ImportError:
    pass

try:
    from .hooks import (
        InitVelocitiesOnConverge,
        ProgressHook,
        StatusTransitionLogger,
        make_graph_tagged_writer,
        make_safety_hooks,
        stdout_writer,
    )
except ImportError:
    pass

try:
    from .io import (
        _restore_arrays,
        _to_jsonable,
        checkpoint_exists,
        extract_per_graph_trajectory,
        fresh_zarr_sink,
        integrator_state_exists,
        load_checkpoint,
        load_integrator_state,
        load_stage_meta,
        load_warmup_csv,
        load_warmup_trajectory,
        load_zarr_frames,
        load_zarr_trajectory,
        next_part_index,
        part_paths,
        read_csv_log,
        save_checkpoint,
        save_integrator_state,
        save_stage_meta,
        zarr_trajectory_length,
    )
except ImportError:
    pass

try:
    from .visualization import (
        dedup_legend,
        plot_batch_speedup,
        plot_slc_stage,
        plot_tm_bracket,
        plot_trajectory_frames,
        plot_warmup_stage,
        shade_stages,
        visualize_structure,
    )
except ImportError:
    pass

try:
    from .notebook_viz import (
        NotebookProgress,
        create_interactive_view,
        display_inline,
        display_trajectory_animation,
        display_widgets_row,
        make_browser_safe_mp4,
        render_trajectory_animation,
        subscript_formula_html,
    )
except ImportError:
    pass

try:
    from .orb import OrbV3Wrapper
except ImportError:
    pass

try:
    from .packmol import (
        build_packmol_slc_stack,
        extract_single_molecule,
        pack_liquid_box,
        pack_with_fixed_obstacle,
    )
except ImportError:
    pass

__all__ = [
    "AMU_OVER_A3_TO_G_CM3",
    "Config",
    "D3_PRESETS",
    "NotebookProgress",
    "create_interactive_view",
    "display_inline",
    "display_trajectory_animation",
    "display_widgets_row",
    "make_browser_safe_mp4",
    "render_trajectory_animation",
    "subscript_formula_html",
    "DYNAMICS_SCALARS",
    "InitVelocitiesOnConverge",
    "MAX_FORCE_CLAMP",
    "OrbV3Wrapper",
    "P_1ATM",
    "STAGE_COLORS",
    "STAGE_LABELS",
    "StatusTransitionLogger",
    "_mol_inertia_eigvecs",
    "_restore_arrays",
    "_to_jsonable",
    "batch_to_ase",
    "build_packmol_slc_stack",
    "checkpoint_exists",
    "compute_S0_from_frames",
    "compute_S0_tail",
    "compute_com_msd",
    "compute_com_msd_numpy",
    "compute_density",
    "compute_mol_axes",
    "compute_molecule_axes",
    "compute_msd",
    "compute_rACF",
    "compute_rdf",
    "compute_rotational_acf",
    "dedup_legend",
    "density_scalar",
    "extract_per_graph_trajectory",
    "extract_single_molecule",
    "fit_diffusion_coefficient",
    "fresh_zarr_sink",
    "integrator_state_exists",
    "load_checkpoint",
    "load_integrator_state",
    "load_stage_meta",
    "load_warmup_csv",
    "load_warmup_trajectory",
    "load_zarr_frames",
    "load_zarr_trajectory",
    "zarr_trajectory_length",
    "ProgressHook",
    "make_graph_tagged_writer",
    "make_safety_hooks",
    "min_pbc_distance",
    "next_part_index",
    "plot_batch_speedup",
    "pack_liquid_box",
    "pack_with_fixed_obstacle",
    "part_paths",
    "plot_slc_stage",
    "plot_tm_bracket",
    "plot_trajectory_frames",
    "plot_warmup_stage",
    "pressure_scalar",
    "read_csv_log",
    "save_checkpoint",
    "save_integrator_state",
    "save_stage_meta",
    "shade_stages",
    "status_by_stage",
    "stdout_writer",
    "visualize_structure",
    "volume_scalar",
    "warmup_stage_names",
]
