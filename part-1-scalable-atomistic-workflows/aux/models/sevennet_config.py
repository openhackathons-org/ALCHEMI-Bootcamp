"""Pinned SevenNet-Omni and PBE-D3(BJ) settings for Part 1.

The model metadata comes from SevenNet's official pretrained-model registry.
The PBE-D3(BJ) parameters are the values documented by the Toolkit
``DFTD3ModelWrapper`` and the DFT-D3 reference implementation.  Its reference
cutoff is 95 bohr with no taper.
"""

from __future__ import annotations


SEVENNET_PACKAGE_VERSION = "0.13.0"
SEVENNET_MODEL_NAME = "7net-omni"
SEVENNET_MODALITY = "mpa"
SEVENNET_REFERENCE_METHOD = "PBE(+U) multi-dataset task; no D3 term"
SEVENNET_CHECKPOINT_URL = (
    "https://github.com/MDIL-SNU/SevenNet/releases/download/"
    "v0.12.0.cp/checkpoint_sevennet_omni.pth"
)
SEVENNET_CHECKPOINT_SHA256 = (
    "ca81bd3aac9fc2696c93dd386615f5a0fe41b92ab9ed7f69fa9526baaa9bab64"
)
SEVENNET_CHECKPOINT_BYTES = 103_162_838
SEVENNET_CHECKPOINT_DOI = "10.6084/m9.figshare.30399814"

# Repeat-call checks catch broken graph ownership or output slicing.  These are
# numerical implementation tolerances, not model-accuracy claims.
SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM = 5.0e-5
SEVENNET_REPEAT_FORCE_TOL_EV_A = 5.0e-4

# PBE-D3(BJ), Grimme et al.  Parameters a2 and the reference cutoff are in
# bohr in the original model.  DFTD3ModelWrapper accepts a2 in bohr and its
# cutoff in angstrom.
PBE_D3_BJ_A1 = 0.4289
PBE_D3_BJ_A2_BOHR = 4.4407
PBE_D3_BJ_S8 = 0.7875
PBE_D3_BJ_S6 = 1.0
BOHR_TO_ANGSTROM = 0.529177210544
D3_REFERENCE_CUTOFF_BOHR = 95.0
D3_REFERENCE_CUTOFF_A = D3_REFERENCE_CUTOFF_BOHR * BOHR_TO_ANGSTROM
D3_REFERENCE_SMOOTHING_FRACTION = 0.0


__all__ = [
    "BOHR_TO_ANGSTROM",
    "D3_REFERENCE_CUTOFF_A",
    "D3_REFERENCE_CUTOFF_BOHR",
    "D3_REFERENCE_SMOOTHING_FRACTION",
    "PBE_D3_BJ_A1",
    "PBE_D3_BJ_A2_BOHR",
    "PBE_D3_BJ_S6",
    "PBE_D3_BJ_S8",
    "SEVENNET_CHECKPOINT_BYTES",
    "SEVENNET_CHECKPOINT_DOI",
    "SEVENNET_CHECKPOINT_SHA256",
    "SEVENNET_CHECKPOINT_URL",
    "SEVENNET_MODALITY",
    "SEVENNET_MODEL_NAME",
    "SEVENNET_PACKAGE_VERSION",
    "SEVENNET_REFERENCE_METHOD",
    "SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM",
    "SEVENNET_REPEAT_FORCE_TOL_EV_A",
]
