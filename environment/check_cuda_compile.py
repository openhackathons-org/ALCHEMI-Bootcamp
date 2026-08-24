# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify and prewarm the CUDA compiler path used by Torch and Triton."""

from __future__ import annotations

import torch


def add_one(values: torch.Tensor) -> torch.Tensor:
    """Provide one small public ``torch.compile`` workload."""
    return values + 1


def main() -> None:
    """Compile and execute one CUDA operation through Torch Inductor."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the ALCHEMI tutorial runtime")

    values = torch.zeros(4, device="cuda")
    result = torch.compile(add_one)(values)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.ones_like(values))
    print("CUDA compiler path verified")


if __name__ == "__main__":
    main()
