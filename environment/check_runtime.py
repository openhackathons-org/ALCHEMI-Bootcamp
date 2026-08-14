"""Verify that a command is using the locked v3 tutorial runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = tomllib.loads((ROOT / "environment" / "runtime-pins.toml").read_text())


def direct_url_commit(distribution_name: str) -> str:
    """Return the immutable VCS commit recorded in installed package metadata."""
    distribution = importlib.metadata.distribution(distribution_name)
    direct_url = next(
        (
            file
            for file in distribution.files or ()
            if file.name == "direct_url.json" and ".dist-info" in str(file)
        ),
        None,
    )
    if direct_url is None:
        raise RuntimeError(f"{distribution_name} has no direct_url.json")
    record = json.loads(distribution.locate_file(direct_url).read_text())
    try:
        return str(record["vcs_info"]["commit_id"])
    except KeyError as error:
        raise RuntimeError(
            f"{distribution_name} was not installed from the required Git commit"
        ) from error


def require_equal(label: str, actual: str, expected: str) -> None:
    """Raise with a concise mismatch message."""
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected}, found {actual}")


def require_file_sha256(label: str, path: Path, expected: str) -> None:
    """Require an existing regular file with the pinned SHA-256 digest."""
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    require_equal(f"{label} SHA-256", actual, expected)


def main() -> None:
    """Check Python, package versions, VCS commits, and runtime assets."""
    require_equal(
        "Python", f"{sys.version_info.major}.{sys.version_info.minor}", PINS["python"]
    )

    for section in ("toolkit", "toolkit_ops"):
        pin = PINS[section]
        name = pin["distribution"]
        require_equal(
            f"{name} version", importlib.metadata.version(name), pin["version"]
        )
        require_equal(f"{name} commit", direct_url_commit(name), pin["commit"])

    for name, expected in (
        ("torch", PINS["torch"]),
        (PINS["model"]["package"], PINS["model"]["package_version"]),
        (PINS["mace"]["package"], PINS["mace"]["package_version"]),
        (
            PINS["structure_viewer"]["package"],
            PINS["structure_viewer"]["package_version"],
        ),
        ("ase", "3.27.0"),
    ):
        actual = importlib.metadata.version(name).split("+")[0]
        require_equal(f"{name} version", actual, expected)

    runtime_root = Path(sys.prefix).resolve().parent
    for variable in (
        "AIMNET_CACHE_DIR",
        "ALCHEMI_D3_PARAM_FILE",
        "HF_HOME",
        "IPYTHONDIR",
        "JUPYTER_CONFIG_DIR",
        "JUPYTER_DATA_DIR",
        "MPLCONFIGDIR",
        "RUFF_CACHE_DIR",
        "TORCH_HOME",
        "UV_CACHE_DIR",
        "WARP_CACHE_PATH",
        "XDG_CACHE_HOME",
    ):
        value = Path(os.environ[variable]).resolve()
        if runtime_root not in (value, *value.parents):
            raise RuntimeError(
                f"{variable} is outside the shared runtime root: {value}"
            )

    require_file_sha256(
        "D3 parameter file",
        Path(os.environ["ALCHEMI_D3_PARAM_FILE"]).resolve(),
        PINS["dispersion"]["generated_parameter_sha256"],
    )

    # The structure viewer resolves its bundle from the repository itself rather
    # than from an environment variable: `show_molecule` inlines the JavaScript
    # into its own HTML output, so nothing outside this tree has to be told where
    # the file lives.
    viewer = PINS["structure_viewer"]
    viewer_directory = ROOT / viewer["directory"]
    for filename, expected in (
        ("3Dmol-min.js", viewer["javascript_sha256"]),
        ("3Dmol-min.js.LICENSE.txt", viewer["javascript_notice_sha256"]),
        ("LICENSE", viewer["license_sha256"]),
    ):
        require_file_sha256(
            f"3Dmol.js {filename}", viewer_directory / filename, expected
        )

    checkpoint_name = "aimnet2_wb97m_d3_0.pt"
    checkpoint_path = Path(os.environ["AIMNET_CACHE_DIR"]) / checkpoint_name
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    require_equal(
        f"{checkpoint_name} SHA-256",
        checkpoint_sha256,
        PINS["model"]["checkpoint_sha256"],
    )

    print("v3 runtime verified")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Toolkit: {PINS['toolkit']['commit']}")
    print(f"Toolkit-Ops: {PINS['toolkit_ops']['commit']}")
    print(f"Environment: {Path(sys.prefix).resolve()}")


if __name__ == "__main__":
    main()
