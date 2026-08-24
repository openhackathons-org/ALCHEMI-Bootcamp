# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example molecule construction shared by the Core playbook modules."""

from __future__ import annotations

from ase import Atoms
from ase.build import molecule as ase_molecule
from ase.collections import s22


def load_example_molecule(name: str) -> Atoms:
    """Return a fresh ASE molecule used by the Core modules."""

    if name == "Ammonia":
        structure = ase_molecule("NH3")
    elif name == "Propyne":
        structure = ase_molecule("C3H4_C3v")
    elif name == "Phenol":
        structure = s22["Phenol_dimer"][:13]
    else:
        expected = "Ammonia, Propyne, or Phenol"
        raise ValueError(f"Unknown example molecule {name!r}; expected {expected}")
    return structure.copy()
