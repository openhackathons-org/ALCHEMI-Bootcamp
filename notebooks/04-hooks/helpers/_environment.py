"""Keep known dependency notices out of learner-facing outputs."""

import os
from importlib import import_module

import torch
import tree


def initialize_toolkit_runtime() -> None:
    """Initialize Toolkit quietly when this host has no usable CUDA device."""

    if torch.cuda.is_available():
        return

    # Importing dynamics initializes Warp's CUDA probe even for the CPU LJ model.
    # On a host without a driver, that probe writes expected diagnostics directly
    # to file descriptor 2. Suppress only that known CPU-only initialization.
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 2)
            import_module("nvalchemi.dynamics.hooks")
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def consume_dm_tree_set_notice() -> None:
    """Consume dm-tree's one-time notice before TensorDict logging starts."""

    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 2)
            tree.is_nested(set())
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
