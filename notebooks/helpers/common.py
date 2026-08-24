# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small setup helpers shared by the Core playbook modules."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_RUNTIME_ROOT = _PROJECT_ROOT / ".alchemi-runtime"
_DIRECT_KERNEL_CACHE_PATHS = {
    "AIMNET_CACHE_DIR": _LOCAL_RUNTIME_ROOT / "aimnet",
    "ALCHEMI_D3_PARAM_FILE": _LOCAL_RUNTIME_ROOT / "dftd3" / "dftd3_parameters.pt",
    "HF_HOME": _LOCAL_RUNTIME_ROOT / "huggingface",
    "MPLCONFIGDIR": _LOCAL_RUNTIME_ROOT / "matplotlib",
    "TORCH_HOME": _LOCAL_RUNTIME_ROOT / "torch",
    "TORCHINDUCTOR_CACHE_DIR": _LOCAL_RUNTIME_ROOT / "torch-inductor",
    "TRITON_CACHE_DIR": _LOCAL_RUNTIME_ROOT / "triton",
    "WARP_CACHE_PATH": _LOCAL_RUNTIME_ROOT / "warp",
}


def _configure_local_runtime() -> None:
    """Set cache fallbacks that still apply in a directly selected kernel."""

    for variable, path in _DIRECT_KERNEL_CACHE_PATHS.items():
        os.environ.setdefault(variable, str(path))


def _prime_tree_set_notice() -> None:
    """Consume dm-tree's one-time set notice outside learner-facing cells."""

    import tree

    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 2)
            tree.flatten({"energy", "forces"})
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def configure_tutorial() -> None:
    """Apply the shared tutorial style and silence known library notices."""

    _configure_local_runtime()
    import matplotlib.pyplot as plt

    warnings.filterwarnings(
        "ignore",
        message="Converting a tensor with requires_grad=True to a scalar.*",
        category=UserWarning,
        module=r"nvalchemi\.models\.aimnet2",
    )
    warnings.filterwarnings(
        "ignore",
        message=r'install "ipywidgets".*',
        category=UserWarning,
        module=r"rich\.live",
    )
    plt.style.use(_PROJECT_ROOT / "shared" / "alchemi-dark.mplstyle")
    _prime_tree_set_notice()


def tutorial_device():
    """Choose the workshop device and report the CPU fallback once."""

    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Compute device: {device} · {torch.cuda.get_device_name(device)}")
        return device
    print("Warning: CUDA is unavailable. This walkthrough is running on CPU.")
    return torch.device("cpu")
