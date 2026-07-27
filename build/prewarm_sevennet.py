#!/usr/bin/env python3
"""Download, verify, and smoke-test the Part 1 SevenNet-Omni checkpoint."""

from __future__ import annotations

import argparse
from importlib import metadata
import math
from pathlib import Path
import sys
from collections.abc import Callable
from typing import Any

import numpy as np


try:
    from sevennet_config import (
        SEVENNET_CHECKPOINT_BYTES,
        SEVENNET_CHECKPOINT_SHA256,
        SEVENNET_CHECKPOINT_URL,
        SEVENNET_MODALITY,
        SEVENNET_MODEL_NAME,
        SEVENNET_PACKAGE_VERSION,
        SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM,
        SEVENNET_REPEAT_FORCE_TOL_EV_A,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = (
        repo_root
        / "part-1-scalable-atomistic-workflows"
        / "aux"
        / "models"
    )
    sys.path.insert(0, str(config_dir))
    from sevennet_config import (  # type: ignore[no-redef]  # noqa: E402
        SEVENNET_CHECKPOINT_BYTES,
        SEVENNET_CHECKPOINT_SHA256,
        SEVENNET_CHECKPOINT_URL,
        SEVENNET_MODALITY,
        SEVENNET_MODEL_NAME,
        SEVENNET_PACKAGE_VERSION,
        SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM,
        SEVENNET_REPEAT_FORCE_TOL_EV_A,
    )


def _part1_root(source_root: Path | None = None) -> Path:
    """Return the Part 1 source directory used by the custom adapter smoke."""

    root = (
        Path(__file__).resolve().parents[1]
        if source_root is None
        else source_root.resolve()
    )
    part1_root = root / "part-1-scalable-atomistic-workflows"
    required = (
        part1_root / "aux" / "models" / "sevennet.py",
        part1_root / "aux" / "models" / "sevennet_checkpoint.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "custom SevenNet preflight source is incomplete: " + ", ".join(missing)
        )
    return part1_root


def _smoke_atoms():
    """Build the periodic Cu(111)+CO input shared by both implementations."""

    from ase.build import add_adsorbate, fcc111, molecule

    atoms = fcc111("Cu", size=(2, 2, 3), vacuum=8.0)
    atoms.pbc = (True, True, False)
    add_adsorbate(
        atoms,
        molecule("CO"),
        height=1.85,
        position="ontop",
        mol_index=1,
    )
    return atoms


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""

    from hashlib import sha256

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(
    *,
    version_reader: Callable[[str], str] | None = None,
    sevenn_constants: Any | None = None,
    path_resolver: Callable[[str], str | Path] | None = None,
    expected_bytes: int | None = None,
    digest_reader: Callable[[Path], str] = sha256_file,
) -> tuple[Path, str]:
    """Download the official checkpoint and verify its identity."""

    if version_reader is None:
        version_reader = metadata.version
    if sevenn_constants is None:
        from sevenn import _const

        sevenn_constants = _const
    if path_resolver is None:
        from sevenn.util import pretrained_name_to_path

        path_resolver = pretrained_name_to_path
    if expected_bytes is None:
        expected_bytes = SEVENNET_CHECKPOINT_BYTES

    installed_version = version_reader("sevenn")
    if installed_version != SEVENNET_PACKAGE_VERSION:
        raise RuntimeError(
            f"expected sevenn {SEVENNET_PACKAGE_VERSION}, found {installed_version}"
        )

    upstream_url = sevenn_constants.CHECKPOINT_DOWNLOAD_LINKS.get(
        sevenn_constants.SEVENNET_omni
    )
    if upstream_url != SEVENNET_CHECKPOINT_URL:
        raise RuntimeError(
            "sevenn's 7net-omni asset URL changed: "
            f"expected {SEVENNET_CHECKPOINT_URL}, found {upstream_url}"
        )

    path = Path(path_resolver(SEVENNET_MODEL_NAME)).resolve()
    if path.name != "checkpoint_sevennet_omni.pth":
        raise RuntimeError(f"unexpected SevenNet checkpoint filename: {path.name!r}")
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(
            "SevenNet-Omni checkpoint size mismatch: "
            f"{path.stat().st_size} != {expected_bytes}"
        )
    digest = digest_reader(path)
    if digest != SEVENNET_CHECKPOINT_SHA256:
        raise RuntimeError(
            "SevenNet-Omni checkpoint SHA-256 mismatch: "
            f"{digest} != {SEVENNET_CHECKPOINT_SHA256}"
        )
    return path, digest


def _official_calculator_smoke(
    checkpoint_path: Path,
    *,
    device: str,
) -> tuple[dict[str, object], np.ndarray]:
    """Evaluate Cu(111)+CO through SevenNet's official ASE calculator."""

    from sevenn import _keys as key
    from sevenn.calculator import SevenNetCalculator
    from sevenn.util import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path)
    modal_map = checkpoint.config.get(key.MODAL_MAP, {})
    if SEVENNET_MODALITY not in modal_map:
        raise RuntimeError(
            f"{SEVENNET_MODEL_NAME} does not expose {SEVENNET_MODALITY!r}; "
            f"available tasks: {sorted(modal_map)}"
        )
    cutoff = float(checkpoint.config[key.CUTOFF])
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise RuntimeError(f"invalid SevenNet cutoff in checkpoint: {cutoff}")

    type_map = checkpoint.config[key.TYPE_MAP]
    required_atomic_numbers = {1, 6, 7, 8, 29}
    missing = sorted(required_atomic_numbers.difference(type_map))
    if missing:
        raise RuntimeError(
            f"SevenNet checkpoint does not cover required atomic numbers: {missing}"
        )

    atoms = _smoke_atoms()
    calculator = SevenNetCalculator(
        model=str(checkpoint_path),
        file_type="checkpoint",
        device=device,
        modal=SEVENNET_MODALITY,
        enable_cueq=False,
        enable_flash=False,
        enable_oeq=False,
        compute_atomic_virial=False,
    )
    try:
        atoms.calc = calculator
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(
            atoms.get_forces(apply_constraint=False),
            dtype=float,
        ).copy()
    finally:
        atoms.calc = None
    if not math.isfinite(energy):
        raise RuntimeError("SevenNet smoke-test energy is not finite")
    if forces.shape != (len(atoms), 3) or not bool(np.isfinite(forces).all()):
        raise RuntimeError(
            f"SevenNet smoke-test forces are invalid: shape={forces.shape}"
        )

    summary: dict[str, object] = {
        "device": device,
        "task": SEVENNET_MODALITY,
        "cutoff_A": cutoff,
        "atoms": len(atoms),
        "energy_eV": energy,
        "force_shape": tuple(forces.shape),
        "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
    }
    return summary, forces


def toolkit_wrapper_smoke(
    checkpoint_path: Path,
    *,
    device: str,
    source_root: Path | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    """Evaluate the same structure through the Part 1 Toolkit adapter."""

    import torch
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.neighbors import compute_neighbors

    part1_root = _part1_root(source_root)
    part1_text = str(part1_root)
    if part1_text not in sys.path:
        sys.path.insert(0, part1_text)

    from aux.models.sevennet import SevenNetOmniWrapper
    from aux.models.sevennet_checkpoint import load_raw_sevennet_omni

    torch_device = torch.device(device)
    raw_model, _ = load_raw_sevennet_omni(
        checkpoint_path,
        device=torch_device,
    )
    wrapper = SevenNetOmniWrapper(
        raw_model,
        modality=SEVENNET_MODALITY,
    ).to(torch_device)
    wrapper.eval()

    atoms = _smoke_atoms()
    data = AtomicData.from_atoms(
        atoms,
        device=torch_device,
        dtype=torch.float32,
    )
    batch = Batch.from_data_list([data], device=torch_device)
    compute_neighbors(batch, config=wrapper.model_config.neighbor_config)

    raw_output_types: list[str] = []

    def record_raw_output(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        raw_output_types.append(
            f"{type(output).__module__}.{type(output).__qualname__}"
        )

    handle = raw_model.register_forward_hook(record_raw_output)
    try:
        outputs = wrapper(batch)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
    finally:
        handle.remove()

    if raw_output_types != ["sevenn.atom_graph_data.AtomGraphData"]:
        raise RuntimeError(
            "unexpected raw SevenNet output container: "
            f"{raw_output_types or ['no output captured']}"
        )

    energy_tensor = outputs.get("energy")
    force_tensor = outputs.get("forces")
    if not isinstance(energy_tensor, torch.Tensor) or tuple(energy_tensor.shape) != (
        1,
        1,
    ):
        raise RuntimeError(
            "Toolkit SevenNet smoke-test energy has the wrong type or shape"
        )
    if not isinstance(force_tensor, torch.Tensor) or tuple(force_tensor.shape) != (
        len(atoms),
        3,
    ):
        raise RuntimeError(
            "Toolkit SevenNet smoke-test forces have the wrong type or shape"
        )
    if not bool(torch.isfinite(energy_tensor).all()):
        raise RuntimeError("Toolkit SevenNet smoke-test energy is not finite")
    if not bool(torch.isfinite(force_tensor).all()):
        raise RuntimeError("Toolkit SevenNet smoke-test forces are not finite")

    energy = float(energy_tensor.detach().reshape(()).cpu())
    forces = force_tensor.detach().cpu().numpy().astype(float, copy=True)
    summary = {
        "device": device,
        "task": SEVENNET_MODALITY,
        "atoms": len(atoms),
        "directed_edges": int(batch.neighbor_list.shape[0]),
        "raw_output_type": raw_output_types[0],
        "energy_eV": energy,
        "force_shape": tuple(forces.shape),
        "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
    }
    return summary, forces


def smoke_test(
    checkpoint_path: Path,
    *,
    device: str,
    source_root: Path | None = None,
    official_smoke: Callable[..., tuple[dict[str, object], np.ndarray]] | None = None,
    toolkit_smoke: Callable[..., tuple[dict[str, object], np.ndarray]] | None = None,
) -> dict[str, object]:
    """Compare the official calculator and the real Part 1 Toolkit adapter."""

    if official_smoke is None:
        official_smoke = _official_calculator_smoke
    if toolkit_smoke is None:
        toolkit_smoke = toolkit_wrapper_smoke

    official, official_forces = official_smoke(
        checkpoint_path,
        device=device,
    )
    toolkit, toolkit_forces = toolkit_smoke(
        checkpoint_path,
        device=device,
        source_root=source_root,
    )
    if toolkit_forces.shape != official_forces.shape:
        raise RuntimeError(
            "official and Toolkit SevenNet force arrays have different shapes"
        )

    atoms = int(toolkit["atoms"])
    energy_difference = abs(
        float(toolkit["energy_eV"]) - float(official["energy_eV"])
    )
    energy_difference_per_atom = energy_difference / atoms
    force_difference = float(np.max(np.abs(toolkit_forces - official_forces)))
    agreement = {
        "energy_difference_eV": energy_difference,
        "energy_difference_eV_per_atom": energy_difference_per_atom,
        "max_force_component_difference_eV_A": force_difference,
        "energy_tolerance_eV_per_atom": (
            SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM
        ),
        "force_tolerance_eV_A": SEVENNET_REPEAT_FORCE_TOL_EV_A,
    }
    if energy_difference_per_atom >= SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM:
        raise RuntimeError(
            "Toolkit SevenNet energy does not match the official calculator: "
            f"{agreement}"
        )
    if force_difference >= SEVENNET_REPEAT_FORCE_TOL_EV_A:
        raise RuntimeError(
            "Toolkit SevenNet forces do not match the official calculator: "
            f"{agreement}"
        )

    return {
        "official_calculator": official,
        "toolkit_wrapper": toolkit,
        "agreement": agreement,
    }


def cpu_smoke(
    checkpoint_path: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, object]:
    """Run the checkpoint smoke test on CPU for image builds and local checks."""

    return smoke_test(
        checkpoint_path,
        device="cpu",
        source_root=source_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--remove-checkpoint-after-test",
        action="store_true",
        help="Delete the downloaded checkpoint after the smoke test.",
    )
    args = parser.parse_args()

    checkpoint_path, digest = resolve_checkpoint()
    smoke = smoke_test(
        checkpoint_path,
        device=args.device,
        source_root=args.source_root,
    )
    print(f"sevenn: {SEVENNET_PACKAGE_VERSION}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint SHA-256: {digest}")
    print(f"official calculator smoke: {smoke['official_calculator']}")
    print(f"custom Toolkit adapter smoke: {smoke['toolkit_wrapper']}")
    print(f"official/Toolkit agreement: {smoke['agreement']}")
    if args.remove_checkpoint_after_test:
        checkpoint_path.unlink()
        print("checkpoint removed after validation")


if __name__ == "__main__":
    main()
