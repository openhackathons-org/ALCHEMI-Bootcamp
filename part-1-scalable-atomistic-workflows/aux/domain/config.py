"""Versioned methodology settings for the Part 1 domain lesson.

This module is the single source for values that change the molecular-box,
electrostatics, dispersion, domain-decomposition, parity, or evaluation
methodology.  It uses only the Python standard library so the campaign planner
can show its defaults on a machine without ASE, Torch, CUDA, or Toolkit.

Callers may still override campaign settings through explicit command-line
arguments.  The planner and notebook record those resolved values separately
from this source configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any


DOMAIN_METHODOLOGY_SCHEMA = "alchemi.part1-domain-methodology.v2"


def _setting(
    *,
    units: str,
    scope: str,
    rationale: str,
    source: str,
) -> Any:
    """Declare the required descriptive metadata for one setting."""

    return field(
        metadata={
            "units": units,
            "scope": scope,
            "rationale": rationale,
            "source": source,
        }
    )


@dataclass(frozen=True)
class DomainMethodologyConfig:
    """One complete, named version of the domain-lesson methodology."""

    name: str
    version: str
    nci_system_id: str = _setting(
        units="NCI Atlas system identifier",
        scope="phenol/N-methylacetamide molecular templates",
        rationale=(
            "Selects the same hydrogen-bonded phenol/N-methylacetamide source "
            "fragments used in Stage 3."
        ),
        source="Packaged Part 1 NCI Atlas subset.",
    )
    nci_scale: float = _setting(
        units="R/Re",
        scope="phenol/N-methylacetamide molecular templates",
        rationale="Selects the reference-scale AB, A, and B records.",
        source="Packaged Part 1 NCI Atlas subset.",
    )
    atoms_per_composition_unit: int = _setting(
        units="atoms per 1:1 phenol/N-methylacetamide unit",
        scope="planned atom counts and saved-result identity",
        rationale=(
            "Records the atom count implied by the selected NCI Atlas "
            "templates so planners and result checks use the same value."
        ),
        source="Packaged Part 1 NCI Atlas subset, system 1.041.",
    )
    aimnet_neighbor_cutoff_a: float = _setting(
        units="Å",
        scope="AIMNet2 neighbor list and DomainParallel halo",
        rationale=(
            "Records the fixed checkpoint cutoff used when deriving the "
            "largest model cutoff for domain decomposition."
        ),
        source="aimnet2-wb97m-d3_0 checkpoint metadata.",
    )
    live_molecules_per_species: int = _setting(
        units="molecules/species",
        scope="live notebook API check",
        rationale=(
            "Keeps the live one-GPU DomainParallel example at 3,200 atoms "
            "while using the same checked 1:1 base box as the campaign."
        ),
        source="Part 1 domain lesson design, declared before the recorded campaign.",
    )
    capacity_molecules_per_species: tuple[int, ...] = _setting(
        units="molecules/species",
        scope="one-GPU capacity ladder",
        rationale=(
            "Uses a declared doubling ladder with no atom-count or memory-fit "
            "skip; the first genuine CUDA OOM remains a measured row."
        ),
        source="Part 1 domain campaign plan, declared before measurement.",
    )
    electrostatics_validation_molecules_per_species: int = _setting(
        units="molecules/species",
        scope="fixed-charge PME-versus-Ewald check",
        rationale=(
            "Checks the electrostatics settings on the smallest planned "
            "periodic box before the capacity ladder."
        ),
        source="Part 1 domain campaign plan, declared before measurement.",
    )
    parity_molecules_per_species: int = _setting(
        units="molecules/species",
        scope="1/2/4-GPU parity input",
        rationale=(
            "Uses the 2 x 2 x 4 repeat of the checked base box. A direct "
            "Toolkit halo check predicts remote atoms on every rank at two and "
            "four GPUs; require_nondegenerate verifies this during each run."
        ),
        source="Part 1 domain decomposition geometry check.",
    )
    construction_density_g_cm3: float = _setting(
        units="g/cm^3",
        scope="Packmol box-volume construction",
        rationale=(
            "Provides a reproducible starting volume; it is a construction "
            "density, not an equilibrated or predicted material density."
        ),
        source="Part 1 Packmol starting-geometry methodology.",
    )
    packmol_tolerance_a: float = _setting(
        units="Å",
        scope="Packmol intermolecular placement",
        rationale="Sets the requested minimum intermolecular placement distance.",
        source="Part 1 Packmol starting-geometry methodology.",
    )
    packmol_precision_a: float = _setting(
        units="Å",
        scope="Packmol optimizer and periodic distance validation",
        rationale=(
            "Keeps optimizer precision well below the placement tolerance and "
            "defines the validation allowance."
        ),
        source="Part 1 Packmol starting-geometry methodology.",
    )
    packmol_seed: int = _setting(
        units="integer seed",
        scope="checked base-box Packmol placement",
        rationale=(
            "Reproduces the one saved 128-pair placement from which every "
            "larger deterministic supercell is built."
        ),
        source="Part 1 Packmol starting-geometry methodology.",
    )
    pme_realspace_cutoff_a: float = _setting(
        units="Å",
        scope="PME real-space electrostatics",
        rationale=(
            "Uses the same real-space cutoff in the live, validation, and "
            "distributed periodic calculations."
        ),
        source="Part 1 periodic electrostatics methodology.",
    )
    pme_mesh_safety_factor: float = _setting(
        units="dimensionless multiplier",
        scope="PME reciprocal-grid estimation",
        rationale=(
            "Uses Toolkit-Ops' standard PME mesh estimate without reducing "
            "its recommended grid."
        ),
        source="Toolkit-Ops 0.4 estimate_pme_parameters API.",
    )
    pme_spline_order: int = _setting(
        units="order",
        scope="PME charge interpolation",
        rationale="Keeps charge interpolation identical across campaign cases.",
        source="Part 1 periodic electrostatics methodology.",
    )
    pme_accuracy: float = _setting(
        units="dimensionless solver target",
        scope="production PME parameter estimation",
        rationale=(
            "Derives a mutually consistent PME splitting parameter and mesh "
            "for the declared real-space cutoff."
        ),
        source="Toolkit-Ops 0.4 estimate_pme_parameters API.",
    )
    ewald_reference_accuracy: float = _setting(
        units="dimensionless solver target",
        scope="fixed-charge Ewald reference",
        rationale=(
            "Builds a direct Ewald reference with a tighter target than the "
            "production PME while keeping its estimated real-space cutoff "
            "below half the validation-box length."
        ),
        source="Toolkit-Ops 0.4 estimate_ewald_parameters API.",
    )
    pme_ewald_energy_tolerance_ev_per_atom: float = _setting(
        units="eV/atom",
        scope="fixed-charge PME-versus-Ewald acceptance",
        rationale="Checks equivalent periodic electrostatics routes per atom.",
        source="Part 1 acceptance limit declared before measurement.",
    )
    pme_ewald_force_max_tolerance_ev_a: float = _setting(
        units="eV/Å",
        scope="fixed-charge PME-versus-Ewald acceptance",
        rationale=(
            "Limits the largest atomic force-vector difference between the two "
            "electrostatics routes."
        ),
        source="Part 1 acceptance limit declared before measurement.",
    )
    charge_sum_tolerance_e: float = _setting(
        units="e",
        scope="one-GPU predicted-charge neutrality",
        rationale="Catches a missing, non-finite, or non-neutral charge result.",
        source="Part 1 numerical acceptance limit declared before measurement.",
    )
    parity_energy_tolerance_ev_per_atom: float = _setting(
        units="eV/atom",
        scope="2-GPU versus 4-GPU same-input energy agreement",
        rationale=(
            "Checks consistency across the distributed reduction path and "
            "normalizes by atom count so the atomic-energy baseline does not "
            "set the acceptance limit."
        ),
        source="Part 1 numerical acceptance limit declared before measurement.",
    )
    parity_force_atol_ev_a: float = _setting(
        units="eV/Å",
        scope="1-GPU versus multi-GPU componentwise force parity",
        rationale="Provides the absolute term in the declared force parity rule.",
        source="Toolkit 0.2 AIMNet2-PME distributed-composite regression methodology.",
    )
    parity_force_rtol: float = _setting(
        units="fraction",
        scope="1-GPU versus multi-GPU componentwise force parity",
        rationale="Allows the declared force limit to scale by component.",
        source="Toolkit 0.2 AIMNet2-PME distributed-composite regression methodology.",
    )
    d3_cutoff_a: float = _setting(
        units="Å",
        scope="D3(BJ) dispersion",
        rationale="Matches the smoothed finite-cutoff D3 setup used in Stage 3.",
        source="Part 1 Stage 3 D3 methodology.",
    )
    d3_smoothing_fraction: float = _setting(
        units="fraction of cutoff",
        scope="D3(BJ) cutoff smoothing",
        rationale="Matches the D3 cutoff smoothing used in Stage 3.",
        source="Part 1 Stage 3 D3 methodology.",
    )
    domain_halo_skin_a: float = _setting(
        units="Å",
        scope="DomainParallel ghost-atom halo",
        rationale=(
            "Adds communication depth for D3 coordination numbers around "
            "borrowed atoms; acceptance still requires same-input force parity."
        ),
        source="Toolkit 0.2 multi-GPU D3 test starting value.",
    )
    steady_timing_warmup_count: int = _setting(
        units="public partition-run-gather workflows/case",
        scope="steady 1/2/4-GPU timing warmup",
        rationale=(
            "Warms the complete public DomainParallel workflow before collecting "
            "steady timing samples."
        ),
        source="Part 1 steady-timing methodology declared before measurement.",
    )
    steady_timing_sample_count: int = _setting(
        units="public partition-run-gather workflows/case",
        scope="steady 1/2/4-GPU timing samples",
        rationale=(
            "Provides enough repeated complete-workflow observations to report "
            "a median and interquartile range."
        ),
        source="Part 1 steady-timing methodology declared before measurement.",
    )
    steady_timing_max_relative_iqr: float = _setting(
        units="IQR / median",
        scope="steady 1/2/4-GPU timing repeatability",
        rationale=(
            "Rejects a tutorial timing series when any GPU count has a broad "
            "middle half of samples. This is a repeatability check, not a "
            "claim of benchmark-grade statistics."
        ),
        source="Part 1 steady-timing methodology declared before measurement.",
    )
    steady_timing_model_evaluations_per_workflow: int = _setting(
        units="model evaluations/public workflow",
        scope="equal-work 1/2/4-GPU steady timing",
        rationale=(
            "Keeps the timed model work identical even though multi-rank "
            "DomainParallel performs an automatic initial force evaluation."
        ),
        source="Part 1 equal-work timing methodology declared before measurement.",
    )
    domain_parallel_multi_rank_initial_force_evaluations: int = _setting(
        units="model evaluations before requested steps",
        scope="pinned Toolkit 0.2 multi-rank DomainParallel.run behavior",
        rationale=(
            "Records the automatic force priming that must be included when "
            "deriving equal-work run-step counts."
        ),
        source=(
            "Toolkit Core commit 331d6b2, "
            "nvalchemi.distributed.domain_parallel.DomainParallel.run."
        ),
    )
    capacity_world_sizes: tuple[int, ...] = _setting(
        units="ranks/GPUs",
        scope="one-GPU capacity phase",
        rationale="Keeps capacity discovery on one fresh single-GPU process per case.",
        source="Part 1 domain campaign execution plan.",
    )
    domain_grid_dims: tuple[int, int, int] | None = _setting(
        units="spatial cells along x/y/z, or automatic",
        scope="DomainConfig.grid_dims",
        rationale=(
            "Uses Toolkit's geometry- and cutoff-based cell grid. This field "
            "does not directly set the rank layout."
        ),
        source="Toolkit 0.2 DomainConfig and SpatialPartitioner API.",
    )
    campaign_world_sizes: tuple[int, ...] = _setting(
        units="ranks/GPUs",
        scope="recorded DomainParallel comparisons",
        rationale=(
            "Runs the recorded comparisons on one, two, and four GPUs. "
            "Toolkit derives a separate rank layout from each input's actual "
            "cell shape."
        ),
        source="Part 1 domain campaign execution plan.",
    )

    def __post_init__(self) -> None:
        if (
            not self.name.strip()
            or not self.version.strip()
            or not self.nci_system_id.strip()
        ):
            raise ValueError(
                "methodology name, version, and NCI system ID must be nonempty"
            )
        integer_names = (
            "atoms_per_composition_unit",
            "live_molecules_per_species",
            "electrostatics_validation_molecules_per_species",
            "parity_molecules_per_species",
            "packmol_seed",
            "pme_spline_order",
            "steady_timing_warmup_count",
            "steady_timing_sample_count",
            "steady_timing_model_evaluations_per_workflow",
            "domain_parallel_multi_rank_initial_force_evaluations",
        )
        for name in integer_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.steady_timing_sample_count < 5:
            raise ValueError("steady_timing_sample_count must be at least 5")
        if self.steady_timing_max_relative_iqr > 1.0:
            raise ValueError("steady_timing_max_relative_iqr must not exceed 1")
        if (
            self.steady_timing_model_evaluations_per_workflow
            <= self.domain_parallel_multi_rank_initial_force_evaluations
        ):
            raise ValueError(
                "steady timing needs at least one requested multi-rank step"
            )
        counts = self.capacity_molecules_per_species
        if (
            not isinstance(counts, tuple)
            or not counts
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in counts
            )
            or tuple(sorted(set(counts))) != counts
        ):
            raise ValueError(
                "capacity_molecules_per_species must be a strictly increasing "
                "tuple of positive integers"
            )
        if self.live_molecules_per_species not in counts:
            raise ValueError("live molecule count must occur in the capacity ladder")
        if self.parity_molecules_per_species not in counts:
            raise ValueError("parity molecule count must occur in the capacity ladder")

        real_names = tuple(
            item.name
            for item in fields(self)
            if item.name not in {"name", "version", "nci_system_id", *integer_names}
            and item.name
            not in {
                "capacity_molecules_per_species",
                "capacity_world_sizes",
                "campaign_world_sizes",
                "d3_smoothing_fraction",
                "domain_grid_dims",
            }
        )
        for name in real_names:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if (
            not math.isfinite(float(self.d3_smoothing_fraction))
            or not 0.0 <= self.d3_smoothing_fraction < 1.0
        ):
            raise ValueError("d3_smoothing_fraction must be in [0, 1)")
        if self.packmol_precision_a >= self.packmol_tolerance_a:
            raise ValueError(
                "packmol_precision_a must be smaller than packmol_tolerance_a"
            )
        if (
            not self.capacity_world_sizes
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.capacity_world_sizes
            )
        ):
            raise ValueError("capacity_world_sizes must contain positive integers")
        if self.domain_grid_dims is not None and (
            len(self.domain_grid_dims) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.domain_grid_dims
            )
        ):
            raise ValueError(
                "domain_grid_dims must be None or three positive integers"
            )
        if (
            not self.campaign_world_sizes
            or len(set(self.campaign_world_sizes)) != len(self.campaign_world_sizes)
            or tuple(sorted(self.campaign_world_sizes)) != self.campaign_world_sizes
            or any(
                isinstance(world_size, bool)
                or not isinstance(world_size, int)
                or world_size <= 0
                for world_size in self.campaign_world_sizes
            )
        ):
            raise ValueError(
                "campaign_world_sizes must be an increasing tuple of positive "
                "integers"
            )
        if any(
            value not in self.campaign_world_sizes
            for value in self.capacity_world_sizes
        ):
            raise ValueError("capacity world sizes must be campaign world sizes")
        if self.campaign_world_sizes[0] != 1:
            raise ValueError("the recorded campaign requires a one-GPU reference")
        if len(self.distributed_world_sizes) < 2:
            raise ValueError(
                "the recorded campaign requires an energy reference and comparison"
            )

    @property
    def schema(self) -> str:
        return DOMAIN_METHODOLOGY_SCHEMA

    @property
    def supported_world_sizes(self) -> tuple[int, ...]:
        """Return every world size used by the recorded campaign."""

        return self.campaign_world_sizes

    @property
    def distributed_world_sizes(self) -> tuple[int, ...]:
        """Return the multi-rank world sizes used by the recorded campaign."""

        return tuple(
            world_size for world_size in self.supported_world_sizes if world_size > 1
        )

    @property
    def force_reference_world_size(self) -> int:
        """Return the single-GPU reference used for force checks."""

        return 1

    @property
    def force_comparison_world_sizes(self) -> tuple[int, ...]:
        """Return the multi-GPU runs compared with the force reference."""

        return self.distributed_world_sizes

    @property
    def energy_reference_world_size(self) -> int:
        """Return the first distributed run used as the energy reference."""

        return self.distributed_world_sizes[0]

    @property
    def energy_comparison_world_sizes(self) -> tuple[int, ...]:
        """Return distributed runs compared with the energy reference."""

        return self.distributed_world_sizes[1:]

    @property
    def steady_timing_world_sizes(self) -> tuple[int, ...]:
        """Return the 1/2/4-GPU sizes required for steady timing."""

        return self.supported_world_sizes

    def steady_timing_run_steps(self, world_size: int) -> int:
        """Return requested steps that give equal timed model work."""

        if world_size not in self.steady_timing_world_sizes:
            raise ValueError(f"unsupported steady-timing world size: {world_size}")
        automatic = (
            self.domain_parallel_multi_rank_initial_force_evaluations
            if world_size > 1
            else 0
        )
        return self.steady_timing_model_evaluations_per_workflow - automatic

    def resolved_values(self, *, json_compatible: bool = False) -> dict[str, Any]:
        """Return behavior-affecting values without duplicating defaults."""

        values: dict[str, Any] = {}
        for item in fields(self):
            if item.name in {"name", "version"}:
                continue
            value = getattr(self, item.name)
            if json_compatible and isinstance(value, tuple):
                value = list(value)
            values[item.name] = value
        return values

    def as_record(self) -> dict[str, Any]:
        """Return the complete versioned source record for saved outputs."""

        settings: dict[str, dict[str, Any]] = {}
        for item in fields(self):
            if item.name in {"name", "version"}:
                continue
            value = getattr(self, item.name)
            if isinstance(value, tuple):
                value = list(value)
            settings[item.name] = {
                "name": item.name,
                "value": value,
                **dict(item.metadata),
            }
        return {
            "schema": self.schema,
            "name": self.name,
            "version": self.version,
            "settings": settings,
        }

    def table_records(self) -> list[dict[str, Any]]:
        """Return learner-facing rows with value, units, scope, and reasons."""

        return [
            {
                "setting": setting["name"],
                "value": setting["value"],
                "units": setting["units"],
                "scope": setting["scope"],
                "rationale": setting["rationale"],
                "source": setting["source"],
            }
            for setting in self.as_record()["settings"].values()
        ]


DOMAIN_METHODOLOGY = DomainMethodologyConfig(
    name="part1-packmol-domain-decomposition",
    version="1.6.0",
    nci_system_id="1.041",
    nci_scale=1.0,
    atoms_per_composition_unit=25,
    aimnet_neighbor_cutoff_a=5.0,
    live_molecules_per_species=128,
    capacity_molecules_per_species=(
        128,
        256,
        512,
        1_024,
        2_048,
        4_096,
        8_192,
        16_384,
        32_768,
    ),
    electrostatics_validation_molecules_per_species=128,
    parity_molecules_per_species=2_048,
    construction_density_g_cm3=1.0,
    packmol_tolerance_a=2.0,
    packmol_precision_a=1.0e-3,
    packmol_seed=20260723,
    pme_realspace_cutoff_a=12.0,
    pme_mesh_safety_factor=1.0,
    pme_spline_order=4,
    pme_accuracy=1.0e-4,
    ewald_reference_accuracy=2.0e-5,
    pme_ewald_energy_tolerance_ev_per_atom=1.0e-4,
    pme_ewald_force_max_tolerance_ev_a=5.0e-3,
    charge_sum_tolerance_e=1.0e-4,
    parity_energy_tolerance_ev_per_atom=1.0e-4,
    parity_force_atol_ev_a=2.0e-3,
    parity_force_rtol=1.0e-2,
    d3_cutoff_a=15.0,
    d3_smoothing_fraction=0.2,
    domain_halo_skin_a=4.0,
    steady_timing_warmup_count=1,
    steady_timing_sample_count=5,
    steady_timing_max_relative_iqr=0.10,
    steady_timing_model_evaluations_per_workflow=2,
    domain_parallel_multi_rank_initial_force_evaluations=1,
    capacity_world_sizes=(1,),
    domain_grid_dims=None,
    campaign_world_sizes=(1, 2, 4),
)


__all__ = (
    "DOMAIN_METHODOLOGY",
    "DOMAIN_METHODOLOGY_SCHEMA",
    "DomainMethodologyConfig",
)
