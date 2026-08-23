# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for publishing kernel widget assets to the host Jupyter server."""

from __future__ import annotations

from pathlib import Path

import install_widget_frontend
import pytest


def _create_frontend_assets(prefix: Path) -> None:
    lab = (
        prefix
        / "share/jupyter/labextensions/@jupyter-widgets/jupyterlab-manager"
    )
    lab.mkdir(parents=True)
    (lab / "package.json").write_text('{"name":"widget-manager"}\n')
    (lab / "static").mkdir()
    (lab / "static/remoteEntry.js").write_text("widget manager\n")

    classic = prefix / "share/jupyter/nbextensions/jupyter-js-widgets"
    classic.mkdir(parents=True)
    (classic / "extension.js").write_text("classic widgets\n")


def test_install_widget_frontend_publishes_lab_and_classic_assets(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "runtime"
    destination = tmp_path / "host-jupyter"
    _create_frontend_assets(prefix)

    installed = install_widget_frontend.install_widget_frontend(
        environment_prefix=prefix,
        jupyter_data_dir=destination,
    )

    lab = (
        destination
        / "labextensions/@jupyter-widgets/jupyterlab-manager"
    )
    classic = destination / "nbextensions/jupyter-js-widgets"
    assert installed == (lab, classic)
    assert (lab / "package.json").read_text() == '{"name":"widget-manager"}\n'
    assert (lab / "static/remoteEntry.js").read_text() == "widget manager\n"
    assert (classic / "extension.js").read_text() == "classic widgets\n"


def test_install_widget_frontend_refreshes_existing_assets(tmp_path: Path) -> None:
    prefix = tmp_path / "runtime"
    destination = tmp_path / "host-jupyter"
    _create_frontend_assets(prefix)

    target = (
        destination
        / "labextensions/@jupyter-widgets/jupyterlab-manager/static"
    )
    target.mkdir(parents=True)
    (target / "remoteEntry.js").write_text("stale manager\n")

    install_widget_frontend.install_widget_frontend(
        environment_prefix=prefix,
        jupyter_data_dir=destination,
    )

    assert (target / "remoteEntry.js").read_text() == "widget manager\n"


def test_install_widget_frontend_rejects_missing_runtime_assets(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="JupyterLab widget frontend is missing"):
        install_widget_frontend.install_widget_frontend(
            environment_prefix=tmp_path / "empty-runtime",
            jupyter_data_dir=tmp_path / "host-jupyter",
        )
