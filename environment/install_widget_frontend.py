# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish kernel widget assets where the host Jupyter server can find them."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_FRONTENDS = (
    (
        "JupyterLab widget frontend",
        Path("labextensions/@jupyter-widgets/jupyterlab-manager"),
        Path("package.json"),
    ),
    (
        "classic notebook widget frontend",
        Path("nbextensions/jupyter-js-widgets"),
        Path("extension.js"),
    ),
)


def install_widget_frontend(
    *,
    environment_prefix: Path,
    jupyter_data_dir: Path,
) -> tuple[Path, ...]:
    """Copy prebuilt widget frontends from the kernel env to host Jupyter data."""

    source_root = environment_prefix / "share/jupyter"
    installed: list[Path] = []
    for label, relative_path, required_file in _FRONTENDS:
        source = source_root / relative_path
        if not (source / required_file).is_file():
            raise RuntimeError(f"{label} is missing: {source}")

        destination = jupyter_data_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        if not (destination / required_file).is_file():
            raise RuntimeError(f"{label} was not published: {destination}")
        installed.append(destination)

    return tuple(installed)


def main() -> None:
    """Install the locked widget frontends into a host-visible data directory."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment-prefix",
        type=Path,
        default=Path(sys.prefix),
    )
    parser.add_argument("--jupyter-data-dir", type=Path, required=True)
    arguments = parser.parse_args()

    for path in install_widget_frontend(
        environment_prefix=arguments.environment_prefix,
        jupyter_data_dir=arguments.jupyter_data_dir,
    ):
        print(f"Jupyter widget frontend: {path}")


if __name__ == "__main__":
    main()
