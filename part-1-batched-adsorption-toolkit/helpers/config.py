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
"""Notebook run configuration for the batched adsorption tutorial.

The audience-facing notebook builds a single ``Config(...)`` then calls
``cfg.activate()``. All derived paths, environment variables, helper
re-imports and the summary print live behind that interface so tutorial
cells stay focused on the science.
"""

from __future__ import annotations

import inspect
import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .cache_registry import (
    ACCURACY_DIR_NAME,
    CACHE_TABLES_DIR_NAME,
    LABEL_NEW_LIVE_RUN,
    LABEL_OFFICIAL,
    LABEL_OFFICIAL_REFRESH,
    LABEL_SELECTED_LIVE_RUN,
    LATEST_COMPLETE_RUN_ID,
    LIVE_RUN_ROOT,
    OFFICIAL_OUTPUT_ROOT,
    PLOTS_DIR_NAME,
    RUN_ID_FORMAT,
    RUNTIME_CACHE_ROOT,
    SURFACE_SCREEN_DIR_NAME,
    TUTORIAL_DIR_NAME,
    TUTORIAL_SURFACE_SCREEN_STEM,
    find_latest_complete_live_run,
)
from .constants import RUN_SCOPES

_PACKAGE_VERSION_TARGETS: tuple[str, ...] = (
    "torch",
    "nvalchemi-toolkit",
    "nvalchemi-toolkit-ops",
    "mace-torch",
    "ase",
    "pymatgen",
    "pydantic",
    "ovito",
)

_KEY_COLUMN_WIDTH = 33
_SECTION_SEPARATOR = "-" * 50


class Config(BaseModel):
    """User-facing run configuration.

    Only the audience-facing fields are intended to be set by tutorial users;
    the toolkit-acceleration fields have sane defaults and rarely need to
    change. Internal fields (``EXECUTION_PATH``, ``LIVE_RUN_ID``,
    ``TUTORIAL_ROOT``) are computed at instantiation and left out of the
    constructor signature in normal use. Path properties derive directly from
    the active fields, so there is no separate ``RunRoots`` indirection.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Audience-facing fields --------------------------------------------
    RUN_SCOPE: Literal["short", "full"] = "full"
    TUTORIAL_RESULT_SOURCE: Literal["compute", "saved"] = "compute"
    VALIDATION_RESULT_SOURCE: Literal["compute", "saved"] = "compute"
    SAVED_TUTORIAL_RUN_ID: str | None = None
    SAVED_ACCURACY_RUN_ID: str | None = None
    REFRESH_SAVED_RESULTS: bool = False
    REQUIRE_VISRTX_RENDER: bool = False

    # --- Toolkit acceleration (rarely tuned) -------------------------------
    TOOLKIT_DEVICE: str = "auto"
    TOOLKIT_DTYPE: Literal["float32", "float64"] = "float32"
    TOOLKIT_COMPILE_MODEL: bool = False
    TOOLKIT_ENABLE_CUEQ: bool = True
    TOOLKIT_MATMUL_PRECISION: Literal["high", "highest", "medium"] = "high"

    # --- Internals computed at instantiation --------------------------------
    EXECUTION_PATH: Literal["toolkit"] = "toolkit"
    LIVE_RUN_ID: str = Field(
        default_factory=lambda: datetime.now().strftime(RUN_ID_FORMAT)
    )
    TUTORIAL_ROOT: Path = Field(default_factory=lambda: Path.cwd().resolve())

    _activated: bool = PrivateAttr(default=False)
    # Resolved torch objects populated by ``_configure_runtime``. Stored as Any
    # so this module does not need to import torch at the top level.
    _resolved_toolkit_device: Any = PrivateAttr(default=None)
    _resolved_toolkit_dtype: Any = PrivateAttr(default=None)

    # --- Validators --------------------------------------------------------
    @model_validator(mode="after")
    def _apply_refresh_overrides(self) -> "Config":
        if self.REFRESH_SAVED_RESULTS:
            self.TUTORIAL_RESULT_SOURCE = "compute"
            self.VALIDATION_RESULT_SOURCE = "compute"
            self.SAVED_TUTORIAL_RUN_ID = None
            self.SAVED_ACCURACY_RUN_ID = None
        return self

    @model_validator(mode="after")
    def _resolve_latest_complete_run_ids(self) -> "Config":
        if (
            self.USE_SAVED_TUTORIAL_RESULTS
            and self.SAVED_TUTORIAL_RUN_ID == LATEST_COMPLETE_RUN_ID
        ):
            self.SAVED_TUTORIAL_RUN_ID = find_latest_complete_live_run(
                self.TUTORIAL_ROOT,
                run_scope=self.RUN_SCOPE,
                artifact_kind="tutorial",
            )
        if (
            self.VALIDATION_RESULT_SOURCE == "saved"
            and self.SAVED_ACCURACY_RUN_ID == LATEST_COMPLETE_RUN_ID
        ):
            self.SAVED_ACCURACY_RUN_ID = find_latest_complete_live_run(
                self.TUTORIAL_ROOT,
                run_scope=self.RUN_SCOPE,
                artifact_kind="accuracy",
            )
        return self

    # --- Simple derived values ---------------------------------------------
    @property
    def RUN_SCOPE_LABEL(self) -> str:
        return RUN_SCOPES[self.RUN_SCOPE]

    @property
    def USE_SAVED_TUTORIAL_RESULTS(self) -> bool:
        return self.TUTORIAL_RESULT_SOURCE == "saved"

    # --- Path-derivation primitives ----------------------------------------
    def _relative(self, path: Path) -> Path:
        """Return *path* relative to TUTORIAL_ROOT when possible, else absolute."""
        try:
            return path.resolve().relative_to(self.TUTORIAL_ROOT.resolve())
        except ValueError:
            return path

    @property
    def _official_tutorial_path(self) -> Path:
        return self.TUTORIAL_ROOT / OFFICIAL_OUTPUT_ROOT / TUTORIAL_DIR_NAME

    @property
    def _official_accuracy_path(self) -> Path:
        return self.TUTORIAL_ROOT / OFFICIAL_OUTPUT_ROOT / ACCURACY_DIR_NAME

    @property
    def _live_path(self) -> Path:
        return self.TUTORIAL_ROOT / LIVE_RUN_ROOT / self.LIVE_RUN_ID

    @property
    def _tutorial_resolved_path(self) -> Path:
        """Where the tutorial section reads or writes its outputs."""
        if self.REFRESH_SAVED_RESULTS:
            return self._official_tutorial_path
        if self.USE_SAVED_TUTORIAL_RESULTS:
            if self.SAVED_TUTORIAL_RUN_ID:
                return (
                    self.TUTORIAL_ROOT
                    / LIVE_RUN_ROOT
                    / self.SAVED_TUTORIAL_RUN_ID
                    / TUTORIAL_DIR_NAME
                )
            return self._official_tutorial_path
        return self._live_path / TUTORIAL_DIR_NAME

    @property
    def _accuracy_resolved_path(self) -> Path:
        """Where the accuracy section reads or writes its outputs."""
        if self.REFRESH_SAVED_RESULTS:
            return self._official_accuracy_path
        if self.VALIDATION_RESULT_SOURCE == "saved":
            if self.SAVED_ACCURACY_RUN_ID:
                return (
                    self.TUTORIAL_ROOT
                    / LIVE_RUN_ROOT
                    / self.SAVED_ACCURACY_RUN_ID
                    / ACCURACY_DIR_NAME
                )
            return self._official_accuracy_path
        return self._live_path / ACCURACY_DIR_NAME

    # --- Public path properties (tutorial-root-relative POSIX strings) ------
    @property
    def TUTORIAL_OUTPUT_DIR(self) -> str:
        return self._relative(self._tutorial_resolved_path).as_posix()

    @property
    def ACCURACY_OUTPUT_DIR(self) -> str:
        return self._relative(self._accuracy_resolved_path).as_posix()

    @property
    def CACHE_DIR(self) -> str:
        return self._relative(
            self._tutorial_resolved_path / CACHE_TABLES_DIR_NAME
        ).as_posix()

    @property
    def PLOTS_DIR(self) -> str:
        return self._relative(self._tutorial_resolved_path / PLOTS_DIR_NAME).as_posix()

    @property
    def RUNTIME_CACHE_DIR(self) -> str:
        return RUNTIME_CACHE_ROOT.as_posix()

    @property
    def SURFACE_SCREEN_OUTPUT_ROOT(self) -> str:
        return self._relative(
            self._tutorial_resolved_path
            / f"{TUTORIAL_SURFACE_SCREEN_STEM}_{self.RUN_SCOPE}"
            / SURFACE_SCREEN_DIR_NAME
        ).as_posix()

    @property
    def LIVE_OUTPUT_DIR(self) -> str:
        return self._relative(self._live_path).as_posix()

    @property
    def PRECOMPUTED_TUTORIAL_OUTPUT_DIR(self) -> str:
        return self._relative(self._official_tutorial_path).as_posix()

    @property
    def PRECOMPUTED_ACCURACY_OUTPUT_DIR(self) -> str:
        return self._relative(self._official_accuracy_path).as_posix()

    @property
    def TUTORIAL_SOURCE_LABEL(self) -> str:
        if self.REFRESH_SAVED_RESULTS:
            return LABEL_OFFICIAL_REFRESH
        if self.USE_SAVED_TUTORIAL_RESULTS:
            if self.SAVED_TUTORIAL_RUN_ID:
                return LABEL_SELECTED_LIVE_RUN.format(run_id=self.SAVED_TUTORIAL_RUN_ID)
            return LABEL_OFFICIAL
        return LABEL_NEW_LIVE_RUN.format(run_id=self.LIVE_RUN_ID)

    @property
    def ACCURACY_SOURCE_LABEL(self) -> str:
        if self.REFRESH_SAVED_RESULTS:
            return LABEL_OFFICIAL_REFRESH
        if self.VALIDATION_RESULT_SOURCE == "saved":
            if self.SAVED_ACCURACY_RUN_ID:
                return LABEL_SELECTED_LIVE_RUN.format(run_id=self.SAVED_ACCURACY_RUN_ID)
            return LABEL_OFFICIAL
        return LABEL_NEW_LIVE_RUN.format(run_id=self.LIVE_RUN_ID)

    @property
    def toolkit_device(self) -> Any:
        """Resolved ``torch.device`` — populated by ``_configure_runtime``."""
        assert self._resolved_toolkit_device is not None, (
            "Config.activate() must run before toolkit_device is accessed"
        )
        return self._resolved_toolkit_device

    @property
    def toolkit_dtype(self) -> Any:
        """Resolved ``torch.dtype`` — populated by ``_configure_runtime``."""
        assert self._resolved_toolkit_dtype is not None, (
            "Config.activate() must run before toolkit_dtype is accessed"
        )
        return self._resolved_toolkit_dtype

    @property
    def tutorial_relpath(self):
        """Display helper: render a path relative to ``TUTORIAL_ROOT``."""
        root = Path(self.TUTORIAL_ROOT).resolve()

        def _relpath(path: str | Path) -> str:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                return candidate.resolve().relative_to(root).as_posix()
            except ValueError:
                return candidate.name

        return _relpath

    # --- Side effects ------------------------------------------------------
    def _set_env_vars(self) -> None:
        cache_root = self.TUTORIAL_ROOT / self.RUNTIME_CACHE_DIR
        os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
        os.environ["TORCH_HOME"] = str(cache_root / "torch")
        os.environ["HF_HOME"] = str(cache_root / "hf")
        os.environ["WARP_CACHE_PATH"] = str(cache_root / "warp")
        overwrite_flag = "1" if self.REFRESH_SAVED_RESULTS else "0"
        os.environ["REFRESH_SAVED_RESULTS"] = overwrite_flag
        os.environ["ALCHEMI_ALLOW_ARTIFACT_OVERWRITE"] = overwrite_flag
        os.environ["ALCHEMI_ALLOW_CACHE_OVERWRITE"] = (
            "1" if not self.USE_SAVED_TUTORIAL_RESULTS else overwrite_flag
        )

    def _reload_helpers(self) -> None:
        get_ipython = globals().get("get_ipython")
        ipython = get_ipython() if get_ipython else None
        if ipython is None:
            return
        try:
            ipython.run_line_magic("reload_ext", "autoreload")
            ipython.run_line_magic("autoreload", "2")
        except Exception:
            pass

    def _configure_runtime(self) -> None:
        """Apply notebook-wide warning filters and Torch backend tweaks.

        Same filters and flags as the original "Runtime preflight" cell — the
        audience never has to read about kernel-level workarounds. Runs after
        ``_set_env_vars`` so the first torch import sees the right cache paths.
        """
        import logging
        import warnings

        warnings.filterwarnings(
            "ignore", message=".*torch.jit.script.*", category=DeprecationWarning
        )
        warnings.filterwarnings(
            "ignore",
            message="To copy construct from a tensor.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="The TorchScript type system doesn't support instance-level annotations.*",
            category=UserWarning,
        )

        try:
            import torch
        except ImportError:
            return

        from ._toolkit_compat import apply_toolkit_runtime_compatibility

        apply_toolkit_runtime_compatibility()

        # Logger level + dynamo/JIT-fuser flags must be set *after* torch import:
        # torch's own initialisation otherwise resets the dynamo logger level.
        logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
        try:
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass
        # Disable donated-buffer reuse so torch.compile coexists with MACE's
        # force path, which asks autograd to retain/create graphs.
        try:
            import torch._functorch.config  # noqa: F401

            torch._functorch.config.donated_buffer = False
        except (ImportError, AttributeError):
            pass
        for name in ("_jit_set_texpr_fuser_enabled", "_jit_override_can_fuse_on_gpu"):
            if hasattr(torch._C, name):
                try:
                    getattr(torch._C, name)(False)
                except TypeError:
                    pass

        # Resolve the torch objects the notebook needs at call sites that don't
        # accept bare strings (``torch.zeros(dtype=...)``, ``AtomicData.from_atoms(dtype=...)``).
        if self.TOOLKIT_DEVICE == "auto":
            self._resolved_toolkit_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._resolved_toolkit_device = torch.device(self.TOOLKIT_DEVICE)
        self._resolved_toolkit_dtype = getattr(torch, self.TOOLKIT_DTYPE)
        torch.set_float32_matmul_precision(self.TOOLKIT_MATMUL_PRECISION)

    def _print_summary(self) -> None:
        rows: list[tuple[str, Any]] = [
            ("RUN_SCOPE", f"{self.RUN_SCOPE} ({self.RUN_SCOPE_LABEL})"),
            ("EXECUTION_PATH", self.EXECUTION_PATH),
            ("TUTORIAL_RESULT_SOURCE", self.TUTORIAL_RESULT_SOURCE),
            ("VALIDATION_RESULT_SOURCE", self.VALIDATION_RESULT_SOURCE),
            ("REFRESH_SAVED_RESULTS", self.REFRESH_SAVED_RESULTS),
            ("SAVED_TUTORIAL_RUN_ID", self.SAVED_TUTORIAL_RUN_ID or "official"),
            ("SAVED_ACCURACY_RUN_ID", self.SAVED_ACCURACY_RUN_ID or "official"),
            ("TUTORIAL_OUTPUT_DIR", self.TUTORIAL_OUTPUT_DIR),
            ("ACCURACY_OUTPUT_DIR", self.ACCURACY_OUTPUT_DIR),
            ("CACHE_DIR", self.CACHE_DIR),
            ("SURFACE_SCREEN_OUTPUT_ROOT", self.SURFACE_SCREEN_OUTPUT_ROOT),
            ("TOOLKIT_DEVICE", self.TOOLKIT_DEVICE),
            ("TOOLKIT_DTYPE", self.TOOLKIT_DTYPE),
            ("TOOLKIT_COMPILE_MODEL", self.TOOLKIT_COMPILE_MODEL),
            ("TOOLKIT_ENABLE_CUEQ", self.TOOLKIT_ENABLE_CUEQ),
            ("TOOLKIT_MATMUL_PRECISION", self.TOOLKIT_MATMUL_PRECISION),
            ("REQUIRE_VISRTX_RENDER", self.REQUIRE_VISRTX_RENDER),
        ]
        for key, value in rows:
            print(f"{key:<{_KEY_COLUMN_WIDTH}} : {value}")
        print(_SECTION_SEPARATOR)
        print(f"{'python':<{_KEY_COLUMN_WIDTH}} : {sys.version.split()[0]}")
        for package in _PACKAGE_VERSION_TARGETS:
            try:
                resolved = version(package)
            except PackageNotFoundError:
                resolved = "NOT INSTALLED"
            print(f"{package:<{_KEY_COLUMN_WIDTH}} : {resolved}")

    def EXPORTS(self) -> dict[str, Any]:
        """Legacy ALL_CAPS names that downstream notebook cells still read."""
        return {
            "RUN_SCOPE": self.RUN_SCOPE,
            "RUN_SCOPE_LABEL": self.RUN_SCOPE_LABEL,
            "EXECUTION_PATH": self.EXECUTION_PATH,
            "TUTORIAL_RESULT_SOURCE": self.TUTORIAL_RESULT_SOURCE,
            "VALIDATION_RESULT_SOURCE": self.VALIDATION_RESULT_SOURCE,
            "USE_SAVED_TUTORIAL_RESULTS": self.USE_SAVED_TUTORIAL_RESULTS,
            "SAVED_TUTORIAL_RUN_ID": self.SAVED_TUTORIAL_RUN_ID,
            "SAVED_ACCURACY_RUN_ID": self.SAVED_ACCURACY_RUN_ID,
            "REFRESH_SAVED_RESULTS": self.REFRESH_SAVED_RESULTS,
            "LIVE_RUN_ID": self.LIVE_RUN_ID,
            "REQUIRE_VISRTX_RENDER": self.REQUIRE_VISRTX_RENDER,
            "TUTORIAL_ROOT": self.TUTORIAL_ROOT,
            "TUTORIAL_OUTPUT_DIR": self.TUTORIAL_OUTPUT_DIR,
            "ACCURACY_OUTPUT_DIR": self.ACCURACY_OUTPUT_DIR,
            "CACHE_DIR": self.CACHE_DIR,
            "PLOTS_DIR": self.PLOTS_DIR,
            "RUNTIME_CACHE_DIR": self.RUNTIME_CACHE_DIR,
            "SURFACE_SCREEN_OUTPUT_ROOT": self.SURFACE_SCREEN_OUTPUT_ROOT,
            "LIVE_OUTPUT_DIR": self.LIVE_OUTPUT_DIR,
            "PRECOMPUTED_TUTORIAL_OUTPUT_DIR": self.PRECOMPUTED_TUTORIAL_OUTPUT_DIR,
            "PRECOMPUTED_ACCURACY_OUTPUT_DIR": self.PRECOMPUTED_ACCURACY_OUTPUT_DIR,
            "PRECOMPUTED_OUTPUT_DIR": OFFICIAL_OUTPUT_ROOT.as_posix(),
            "OUTPUT_DIR": self.TUTORIAL_OUTPUT_DIR,
            "ASSETS_DIR": "assets",
            "IMAGES_DIR": os.path.join("assets", "images"),
            "PRESENTATION_PLOTS_DIR": os.path.join(
                self.PRECOMPUTED_TUTORIAL_OUTPUT_DIR, "plots"
            ),
            "TOOLKIT_DEVICE": self.TOOLKIT_DEVICE,
            "TOOLKIT_DTYPE": self.TOOLKIT_DTYPE,
            "TOOLKIT_COMPILE_MODEL": self.TOOLKIT_COMPILE_MODEL,
            "TOOLKIT_ENABLE_CUEQ": self.TOOLKIT_ENABLE_CUEQ,
            "TOOLKIT_MATMUL_PRECISION": self.TOOLKIT_MATMUL_PRECISION,
            # Resolved torch objects populated by Config._configure_runtime.
            "toolkit_device": self.toolkit_device,
            "toolkit_dtype": self.toolkit_dtype,
            "tutorial_relpath": self.tutorial_relpath,
            "CONFIG": self,
        }

    def activate(self, namespace: dict[str, Any] | None = None) -> None:
        """Apply side effects and inject legacy ALL_CAPS names into the caller.

        Idempotent: a second call is a no-op. When ``namespace`` is omitted the
        caller's frame globals are used, which is the notebook namespace for a
        top-level cell. Returns ``None`` so the audience cell does not display
        the full ``Config`` repr alongside the summary print.
        """
        if self._activated:
            return
        self._reload_helpers()
        self._set_env_vars()
        self._configure_runtime()
        self._print_summary()
        if namespace is None:
            namespace = inspect.currentframe().f_back.f_globals
        namespace.update(self.EXPORTS())
        self._activated = True
