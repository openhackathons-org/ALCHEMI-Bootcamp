# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper interface used by Module 00, the Core quick tour."""

from __future__ import annotations

from typing import Any

from .catalog import opening_performance_catalog
from .common import configure_tutorial
from .viewers import show_structure_explorer

__all__ = [
    "configure_tutorial",
    "opening_performance_catalog",
    "plot_relaxation_time_perspective",
    "relax_selected_structures",
    "show_structure_explorer",
]


def relax_selected_structures(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Load the relaxation implementation when the learner starts the run."""

    from .relaxation import relax_selected_structures as implementation

    return implementation(*args, **kwargs)


def plot_relaxation_time_perspective(*args: Any, **kwargs: Any) -> None:
    """Load the plotting implementation when the learner requests the plot."""

    from .relaxation import plot_relaxation_time_perspective as implementation

    implementation(*args, **kwargs)
