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
"""User-facing run configuration for the Part 2 melting-point tutorial.

Mirrors the Part 3 ``helpers/config.py`` pattern: a single pydantic model
bundles the few knobs a student touches together with the frozen parameters of
the canonical cached run. ``activate()`` resolves the torch runtime, prints a
summary + package versions, and injects the legacy ALL-CAPS names
(``DEVICE``, ``TEMPS``, ``RUN_NAME`` ...) that the notebook cells reference, so
downstream cells stay terse and the plumbing stays out of the narrative.

The canonical-run parameters are surfaced in the notebook's configuration cell
(not hidden) so a reader can see exactly what produced the shipped results, but
they are documented as do-not-alter: editing them invalidates the ``"saved"``
cache.
"""

from __future__ import annotations

import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr

_KEYW = 24
_RULE = "─" * 64
_VERSION_TARGETS = ("nvalchemi-toolkit", "torch", "orb-models", "ase", "numpy")


class Config(BaseModel):
    """Run configuration for the melting-point notebook.

    Edit the four run-mode knobs freely; leave the canonical-run parameters at
    their defaults unless you are regenerating the full run yourself.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Run mode (safe to change) ────────────────────────────────────────
    RESULT_SOURCE: Literal["saved", "compute"] = "saved"
    TOOLKIT_DEVICE: str = "auto"  # "auto" | "cuda" | "cpu"
    TOOLKIT_DTYPE: Literal["float32", "float64"] = "float32"
    TOOLKIT_COMPILE_MODEL: bool = False

    # ── Canonical-run parameters (do not alter) ──────────────────────────
    RUN_NAME: str = "naphthalene_orbmol"
    TM_EXP: float = 353.0
    T_WARMUP: float = 200.0
    T_MELT: float = 500.0
    TEMPS: tuple[float, ...] = (200.0, 300.0, 400.0, 500.0)
    SUPERCELL: tuple[int, int, int] = (5, 5, 4)
    MELT_SRC: Literal["npt", "nvt"] = "npt"
    DT: float = 0.5
    FRICTION: float = 0.01
    THERMOSTAT_TIME: float = 100.0
    BAROSTAT_TIME: float = 4000.0
    FMAX: float = 0.15
    FIRE_MAX_STEPS: int = 5000
    WARMUP_NVT_PS: float = 30.0
    WARMUP_NPT_PS: float = 50.0
    MELT_PS: float = 15.0
    SLC_NVT_PS: float = 1.0
    SLC_NPT_PS: float = 300.0
    SNAPSHOT_EVERY: int = 1000
    LOG_EVERY: int = 100
    PROGRESS_EVERY: int = 20  # live progress-bar update cadence (finer than LOG_EVERY)

    # ── Live-demo overrides (compute mode; do not alter) ─────────────────
    # The canonical run thermalised for WARMUP_NVT_PS and minimised up to
    # FIRE_MAX_STEPS; when RESULT_SOURCE="compute" the notebook runs short
    # observable stand-ins so the pipeline is watchable without a long wait.
    LIVE_NVT_PS: float = 1.0  # live NVT burst (vs WARMUP_NVT_PS cached)
    LIVE_FIRE_MAX_STEPS: int = 1000  # live FIRE2 cap (vs FIRE_MAX_STEPS cached)
    ANIM_TARGET_FRAMES: int = 60  # frames in the live OVITO animations (~10 s at fps below)

    _activated: bool = PrivateAttr(default=False)
    _device: Any = PrivateAttr(default=None)
    _dtype: Any = PrivateAttr(default=None)

    # ── Derived values ───────────────────────────────────────────────────
    @property
    def USE_SAVED(self) -> bool:
        return self.RESULT_SOURCE == "saved"

    @property
    def DT_TAG(self) -> str:
        return f"dt{str(self.DT).replace('.', 'p')}fs"  # e.g. 'dt0p5fs'

    @property
    def T_WARMUP_TAG(self) -> str:
        return f"{int(self.T_WARMUP)}k"  # e.g. '200k'

    @property
    def T_MELT_TAG(self) -> str:
        return f"{int(self.T_MELT)}k"  # e.g. '500k'

    @property
    def CACHE_DIR(self) -> Path:
        return Path("data") / "cached" / self.RUN_NAME

    @property
    def CACHE_TRAJ_DIR(self) -> Path:
        return self.CACHE_DIR / "traj"

    @property
    def CACHE_CSV_DIR(self) -> Path:
        return self.CACHE_DIR / "csv"

    @property
    def LOG_DIR(self) -> Path:
        # Live-run scratch sits beside the shipped traj/ and csv/ subdirs under
        # the one run folder, not in a separate top-level logs/ directory.
        return self.CACHE_DIR / "logs"

    # ── Cached-result resolvers (saved path) ─────────────────────────────
    def cached_extxyz(self, stem: str) -> Path | None:
        """Shipped trajectory ``CACHE_TRAJ_DIR/<stem>.extxyz`` (``None`` if absent)."""
        path = self.CACHE_TRAJ_DIR / f"{stem}.extxyz"
        return path if path.exists() else None

    def cached_log_csv(self, stem: str) -> Path | None:
        """Shipped LoggingHook CSV ``CACHE_CSV_DIR/<stem>.csv`` (``None`` if absent)."""
        path = self.CACHE_CSV_DIR / f"{stem}.csv"
        return path if path.exists() else None

    @property
    def device(self) -> Any:
        assert self._device is not None, "Config.activate() must run before .device"
        return self._device

    @property
    def dtype(self) -> Any:
        assert self._dtype is not None, "Config.activate() must run before .dtype"
        return self._dtype

    # ── Side effects ─────────────────────────────────────────────────────
    def _configure_runtime(self) -> None:
        """Resolve device/dtype and apply notebook-wide torch / warning tweaks."""
        # The ASE CIF parser flags monoclinic naphthalene with a cosmetic
        # UserWarning even though COD 2311088 supplies symmetry operators.
        warnings.filterwarnings(
            "ignore",
            message=r"crystal system 'monoclinic' is not interpreted.*",
            category=UserWarning,
        )
        try:
            import torch
        except ImportError:
            return
        if self.TOOLKIT_DEVICE == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.TOOLKIT_DEVICE)
        self._dtype = getattr(torch, self.TOOLKIT_DTYPE)
        torch.set_float32_matmul_precision("high")
        # Disable donated-buffer reuse so torch.compile coexists with the
        # pipeline's retain_graph=True stress autograd (only matters under
        # compile_model=True). Lazy-loaded in newer torch builds.
        try:
            import torch._functorch.config  # noqa: F401

            torch._functorch.config.donated_buffer = False
        except (ImportError, AttributeError):
            pass
        # Toolkit neighbor-list compatibility: the pinned toolkit's
        # NeighborListHook calls neighbor_list(method=None), which is a host-only
        # helper and raises inside a torch context. Inject an explicit method
        # (same runtime shim Part 1 applies).
        try:
            from ._toolkit_compat import apply_toolkit_runtime_compatibility

            apply_toolkit_runtime_compatibility()
        except Exception:
            pass

    def _print_summary(self) -> None:
        rows: list[tuple[str, Any]] = [
            ("RESULT_SOURCE", self.RESULT_SOURCE),
            ("device", self._device),
            ("dtype", self.TOOLKIT_DTYPE),
            ("compile_model", self.TOOLKIT_COMPILE_MODEL),
            ("RUN_NAME", self.RUN_NAME),
            ("TM_EXP (K)", self.TM_EXP),
            ("T_WARMUP (K)", self.T_WARMUP),
            ("T_MELT (K)", self.T_MELT),
            ("TEMPS (K)", ", ".join(str(int(t)) for t in self.TEMPS)),
            ("SUPERCELL", "×".join(map(str, self.SUPERCELL))),
            ("DT (fs)", self.DT),
        ]
        print(_RULE)
        print("Run configuration")
        print(_RULE)
        for key, val in rows:
            print(f"{key:<{_KEYW}}: {val}")
        print(_RULE)
        print(f"{'python':<{_KEYW}}: {sys.version.split()[0]}")
        for pkg in _VERSION_TARGETS:
            try:
                print(f"{pkg:<{_KEYW}}: {version(pkg)}")
            except PackageNotFoundError:
                print(f"{pkg:<{_KEYW}}: (not installed)")
        print(_RULE)

    def EXPORTS(self) -> dict[str, Any]:
        """The ALL-CAPS names injected into the notebook namespace."""
        # Only the live "compute" path writes scratch; don't create an empty
        # logs/ subdir when replaying the shipped cache.
        if not self.USE_SAVED:
            self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        return {
            "DEVICE": self._device,
            "DTYPE": self._dtype,
            "RESULT_SOURCE": self.RESULT_SOURCE,
            "USE_SAVED": self.USE_SAVED,
            "COMPILE_MODEL": self.TOOLKIT_COMPILE_MODEL,
            "RUN_NAME": self.RUN_NAME,
            "TM_EXP": self.TM_EXP,
            "T_WARMUP": self.T_WARMUP,
            "T_MELT": self.T_MELT,
            "TEMPS": self.TEMPS,
            "SUPERCELL": self.SUPERCELL,
            "MELT_SRC": self.MELT_SRC,
            "DT": self.DT,
            "FRICTION": self.FRICTION,
            "THERMOSTAT_TIME": self.THERMOSTAT_TIME,
            "BAROSTAT_TIME": self.BAROSTAT_TIME,
            "FMAX": self.FMAX,
            "FIRE_MAX_STEPS": self.FIRE_MAX_STEPS,
            "WARMUP_NVT_PS": self.WARMUP_NVT_PS,
            "WARMUP_NPT_PS": self.WARMUP_NPT_PS,
            "MELT_PS": self.MELT_PS,
            "SLC_NVT_PS": self.SLC_NVT_PS,
            "SLC_NPT_PS": self.SLC_NPT_PS,
            "SNAPSHOT_EVERY": self.SNAPSHOT_EVERY,
            "LOG_EVERY": self.LOG_EVERY,
            "PROGRESS_EVERY": self.PROGRESS_EVERY,
            "LIVE_NVT_PS": self.LIVE_NVT_PS,
            "LIVE_FIRE_MAX_STEPS": self.LIVE_FIRE_MAX_STEPS,
            "ANIM_TARGET_FRAMES": self.ANIM_TARGET_FRAMES,
            "DT_TAG": self.DT_TAG,
            "T_WARMUP_TAG": self.T_WARMUP_TAG,
            "T_MELT_TAG": self.T_MELT_TAG,
            "CACHE_DIR": self.CACHE_DIR,
            "CACHE_TRAJ_DIR": self.CACHE_TRAJ_DIR,
            "CACHE_CSV_DIR": self.CACHE_CSV_DIR,
            "LOG_DIR": self.LOG_DIR,
        }

    def activate(self, namespace: dict[str, Any] | None = None) -> None:
        """Apply runtime config, print the summary, and export ALL-CAPS names.

        Idempotent: a second call is a no-op. When ``namespace`` is omitted the
        caller's frame globals are used (the notebook namespace for a top-level
        cell). Returns ``None`` so the cell shows the summary, not the repr.
        """
        if self._activated:
            return
        self._configure_runtime()
        self._print_summary()
        if namespace is None:
            import inspect

            namespace = inspect.currentframe().f_back.f_globals
        namespace.update(self.EXPORTS())
        self._activated = True
