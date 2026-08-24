# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CORE_DIR.parent


def _helper_references(notebook_name: str) -> set[str]:
    notebook = json.loads((CORE_DIR / notebook_name).read_text())
    references: set[str] = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        references.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "helpers"
        )
    return references


def test_module_zero_and_one_import_without_pandas() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(CORE_DIR)!r})
from helpers import module00, module01
assert 'pandas' not in sys.modules
assert 'matplotlib.pyplot' not in sys.modules
assert module00.__all__ == [
    'configure_tutorial',
    'opening_performance_catalog',
    'plot_relaxation_time_perspective',
    'relax_selected_structures',
    'show_structure_explorer',
]
assert module01.__all__ == [
    'configure_tutorial',
    'load_example_molecule',
    'show_molecule',
]
module01.load_example_molecule('Ammonia')
module01.show_molecule(module01.load_example_molecule('Ammonia'))
assert 'pandas' not in sys.modules
assert 'matplotlib.pyplot' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_configure_tutorial_sets_direct_kernel_cache_defaults() -> None:
    variables = (
        "AIMNET_CACHE_DIR",
        "ALCHEMI_D3_PARAM_FILE",
        "HF_HOME",
        "MPLCONFIGDIR",
        "TORCH_HOME",
        "WARP_CACHE_PATH",
    )
    code = f"""
import os
import sys
from pathlib import Path
sys.path.insert(0, {str(CORE_DIR)!r})
from helpers.common import configure_tutorial
configure_tutorial()
expected_root = Path({str(PROJECT_ROOT / ".alchemi-runtime")!r})
for variable in {variables!r}:
    assert Path(os.environ[variable]).is_relative_to(expected_root)
"""
    environment = os.environ.copy()
    for variable in variables:
        environment.pop(variable, None)
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def test_module_two_exposes_only_its_notebook_interface() -> None:
    sys.path.insert(0, str(CORE_DIR))
    from helpers import module02

    assert set(module02.__all__) == _helper_references(
        "alchemi-core-02-models-and-simulation.ipynb"
    )


def test_module_three_exposes_only_its_notebook_interface() -> None:
    sys.path.insert(0, str(CORE_DIR))
    from helpers import module03

    assert set(module03.__all__) == _helper_references(
        "alchemi-core-03-scale-a-study.ipynb"
    )


def test_module_three_embeds_the_supplied_scheduling_widget() -> None:
    sys.path.insert(0, str(CORE_DIR))
    from helpers import module03

    html = module03.show_scheduling_widget(total_systems=23).data

    assert '<iframe title="Study scheduling routes"' in html
    assert "srcdoc=" in html
    assert "ALCHEMI_SCHED" in html
    assert "data-runtime-note" in html
    assert 'role="note"' in html
    assert ".alchemi-sched .as-bar" in html
    assert "margin:10px 0" in html
    assert "TOTAL = 23, ACTIVE = 64, CAP = 16" in html
    assert "__ALCHEMI_" not in html


def test_module_three_storyboard_keeps_two_md_stage_routes_comparable() -> None:
    source = (CORE_DIR / "assets" / "03-campaign-scheduling-widget-v1.html").read_text()
    timing_block = source.split("var TIMING = [", 1)[1].split("];", 1)[0]
    timings = [
        (int(nvt), int(nve))
        for nvt, nve in re.findall(r"\[(\d+),(\d+)\]", timing_block)
    ]

    fixed_batch_steps = max(nvt for nvt, _ in timings) + max(nve for _, nve in timings)
    fused_batch_steps = max(nvt + nve for nvt, nve in timings)
    mean_system_steps = sum(nvt + nve for nvt, nve in timings) / len(timings)
    inflight_speedup = fixed_batch_steps / mean_system_steps

    assert fixed_batch_steps == fused_batch_steps
    assert 1.45 <= inflight_speedup <= 1.55
