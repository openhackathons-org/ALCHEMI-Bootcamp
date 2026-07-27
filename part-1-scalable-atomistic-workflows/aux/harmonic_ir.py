"""Finite-difference inputs for a like-for-like harmonic IR comparison.

The functions in this module are deliberately independent of Torch and
ALCHEMI Toolkit.  A caller evaluates the *same complete model* at every
displaced geometry, then passes its forces and dipoles here.  The resulting
arrays have the conventions used by :mod:`aux.reference.core`:

* Cartesian Hessian: ``(3N, 3N)`` in ``Eh / bohr**2``;
* dipole derivative: ``(3N, 3)`` in atomic dipole units per bohr.

In particular, if a model is defined as a checkpoint base plus Coulomb and
D3 terms, all three energy terms must be active in every displaced-geometry
force evaluation.  The dipole finite difference uses the charge or dipole
output that the same model definition actually provides; an energy correction
with no dipole output is not assigned an artificial contribution.  Mixing
force components from different model definitions would not be an
apples-to-apples harmonic comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .reference import solve_vibrational_modes


# CODATA 2018 values, matching the constants used by the reference analysis.
ANGSTROM_PER_BOHR = 0.529_177_210_903
BOHR_PER_ANGSTROM = 1.0 / ANGSTROM_PER_BOHR
EV_PER_HARTREE = 27.211_386_245_988
HARTREE_PER_EV = 1.0 / EV_PER_HARTREE
EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2 = (
    HARTREE_PER_EV * ANGSTROM_PER_BOHR**2
)

# Exact SI constants where defined, plus the CODATA 2018 atomic-mass constant.
# If d(mu)/dQ is expressed as elementary charge / sqrt(u), the double-harmonic
# isotropic IR intensity is this factor times its squared Cartesian norm.
AVOGADRO_PER_MOL = 6.022_140_76e23
ELEMENTARY_CHARGE_C = 1.602_176_634e-19
VACUUM_PERMITTIVITY_F_PER_M = 8.854_187_812_8e-12
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
ATOMIC_MASS_UNIT_KG = 1.660_539_066_60e-27
IR_INTENSITY_KM_MOL_PER_E2_PER_U = (
    AVOGADRO_PER_MOL
    * ELEMENTARY_CHARGE_C**2
    / (
        12.0
        * VACUUM_PERMITTIVITY_F_PER_M
        * SPEED_OF_LIGHT_M_PER_S**2
        * ATOMIC_MASS_UNIT_KG
        * 1_000.0
    )
)


def _readonly(value: np.ndarray, *, dtype: object = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _positive_step(step_angstrom: float) -> float:
    step = float(step_angstrom)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_angstrom must be finite and positive")
    return step


@dataclass(frozen=True)
class CartesianDisplacements:
    """Symmetric Cartesian displacements for one geometry and one step.

    ``plus_angstrom[j]`` and ``minus_angstrom[j]`` displace flattened
    Cartesian coordinate ``j`` by ``+step`` and ``-step``, respectively.
    Thus, a molecule with ``N`` atoms produces ``3N`` structures in each
    array and ``6N`` model evaluations in total.
    """

    geometry_angstrom: np.ndarray
    step_angstrom: float
    plus_angstrom: np.ndarray
    minus_angstrom: np.ndarray
    coordinate_atom_indices: np.ndarray
    coordinate_axes: np.ndarray

    @property
    def n_atoms(self) -> int:
        return int(self.geometry_angstrom.shape[0])

    @property
    def n_coordinates(self) -> int:
        return 3 * self.n_atoms


@dataclass(frozen=True)
class CartesianHessianEstimate:
    """Raw and symmetrized finite-difference Cartesian Hessians."""

    step_angstrom: float
    raw_hessian_eV_per_angstrom2: np.ndarray
    hessian_eV_per_angstrom2: np.ndarray
    raw_hessian_hartree_per_bohr2: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    max_abs_antisymmetry_eV_per_angstrom2: float
    max_abs_antisymmetry_hartree_per_bohr2: float
    max_relative_antisymmetry: float


@dataclass(frozen=True)
class DipoleDerivativeEstimate:
    """Central finite difference of the model's total molecular dipole."""

    step_angstrom: float
    dipoles_plus_e_angstrom: np.ndarray
    dipoles_minus_e_angstrom: np.ndarray
    dipoles_plus_atomic_units: np.ndarray
    dipoles_minus_atomic_units: np.ndarray
    dipole_derivative_3n_by_3_au: np.ndarray


@dataclass(frozen=True)
class HarmonicIRFiniteDifferenceEstimate:
    """Hessian and dipole derivative obtained with the same displacement step."""

    hessian: CartesianHessianEstimate
    dipole_derivative: DipoleDerivativeEstimate

    @property
    def step_angstrom(self) -> float:
        return self.hessian.step_angstrom

    @property
    def hessian_hartree_per_bohr2(self) -> np.ndarray:
        return self.hessian.hessian_hartree_per_bohr2

    @property
    def dipole_derivative_3n_by_3_au(self) -> np.ndarray:
        return self.dipole_derivative.dipole_derivative_3n_by_3_au


@dataclass(frozen=True)
class HarmonicIRModeAnalysis:
    """Projected normal modes and double-harmonic IR stick intensities.

    ``mass_weighted_modes`` contains orthonormal rows in the convention
    ``Q = sqrt(mass_u) * R_bohr``.  The normal-coordinate dipole derivative
    therefore has units ``e / sqrt(u)``.  Translations and rotations have
    already been removed by :func:`aux.reference.solve_vibrational_modes`.
    """

    masses_u: np.ndarray
    frequencies_cm1: np.ndarray
    ir_intensities_km_mol: np.ndarray
    normal_dipole_derivative_modes_by_3_e_per_sqrt_u: np.ndarray
    mass_weighted_modes: np.ndarray
    external_rank: int


@dataclass(frozen=True)
class FiniteDifferenceConvergence:
    """Adjacent-step differences, ordered from coarser to finer steps.

    The summary reports changes but does not choose a "best" step.  A smallest
    step can suffer from numerical noise, so that choice belongs in the
    validation method rather than in this utility.
    """

    steps_angstrom: np.ndarray
    hessian_max_relative_antisymmetry: np.ndarray
    coarse_steps_angstrom: np.ndarray
    fine_steps_angstrom: np.ndarray
    hessian_max_abs_change_hartree_per_bohr2: np.ndarray
    hessian_max_relative_change: np.ndarray
    dipole_derivative_max_abs_change_au: np.ndarray
    dipole_derivative_max_relative_change: np.ndarray


@dataclass(frozen=True)
class HarmonicIRConvergence:
    """Observable changes between adjacent finite-difference steps.

    Rows are ordered from the coarsest to the finest displacement.  Modes use
    increasing projected-Hessian eigenvalue order.  This is appropriate for
    the separated modes of an isolated water monomer.  A small same-index mode
    overlap warns that modes crossed or rotated, in which case individual
    intensities should be compared by mode character or as a subspace.
    """

    finite_difference: FiniteDifferenceConvergence
    steps_angstrom: np.ndarray
    frequencies_cm1: np.ndarray
    ir_intensities_km_mol: np.ndarray
    frequency_abs_change_cm1: np.ndarray
    frequency_max_abs_change_cm1: np.ndarray
    ir_intensity_abs_change_km_mol: np.ndarray
    ir_intensity_relative_change: np.ndarray
    ir_intensity_max_abs_change_km_mol: np.ndarray
    ir_intensity_max_relative_change: np.ndarray
    same_index_mode_squared_overlaps: np.ndarray
    minimum_same_index_mode_squared_overlap: np.ndarray


def symmetric_cartesian_displacements(
    geometry_angstrom: np.ndarray,
    step_angstrom: float,
) -> CartesianDisplacements:
    """Return all ``+/-`` single-coordinate displacements of one molecule."""

    geometry = np.asarray(geometry_angstrom, dtype=np.float64)
    if geometry.ndim != 2 or geometry.shape[1] != 3 or geometry.shape[0] == 0:
        raise ValueError("geometry_angstrom must have shape (n_atoms, 3)")
    if not np.all(np.isfinite(geometry)):
        raise ValueError("geometry_angstrom contains non-finite values")
    step = _positive_step(step_angstrom)

    n_atoms = int(geometry.shape[0])
    n_coordinates = 3 * n_atoms
    plus = np.broadcast_to(geometry, (n_coordinates, n_atoms, 3)).copy()
    minus = plus.copy()
    plus_flat = plus.reshape(n_coordinates, n_coordinates)
    minus_flat = minus.reshape(n_coordinates, n_coordinates)
    coordinate = np.arange(n_coordinates)
    plus_flat[coordinate, coordinate] += step
    minus_flat[coordinate, coordinate] -= step

    return CartesianDisplacements(
        geometry_angstrom=_readonly(geometry),
        step_angstrom=step,
        plus_angstrom=_readonly(plus),
        minus_angstrom=_readonly(minus),
        coordinate_atom_indices=_readonly(coordinate // 3, dtype=np.int64),
        coordinate_axes=_readonly(coordinate % 3, dtype=np.int64),
    )


def molecular_dipoles_from_atomic_predictions(
    positions_angstrom: np.ndarray,
    charges_e: np.ndarray,
    *,
    atomic_dipoles_e_angstrom: np.ndarray | None = None,
    origin_angstrom: np.ndarray | None = None,
    neutral_tolerance_e: float | None = None,
) -> np.ndarray:
    """Combine predicted charges and optional atomic dipoles.

    The point-charge contribution is ``sum_i q_i (r_i - origin)``.  Atomic
    dipole vectors, when supplied, are summed as origin-independent intrinsic
    contributions.  Leading batch dimensions are supported, but the position
    and charge arrays must have identical leading dimensions.  The origin may
    be one shared length-3 vector or one vector per item in those leading batch
    dimensions.

    For a neutral molecule the result is independent of ``origin_angstrom``.
    A charged molecule's dipole is necessarily origin-dependent.  Set
    ``neutral_tolerance_e`` to a non-negative value when neutral predictions
    are required by the scientific method.
    """

    positions = np.asarray(positions_angstrom, dtype=np.float64)
    charges = np.asarray(charges_e, dtype=np.float64)
    if positions.ndim < 2 or positions.shape[-1] != 3:
        raise ValueError("positions_angstrom must have shape (..., n_atoms, 3)")
    if charges.shape != positions.shape[:-1]:
        raise ValueError(
            "charges_e must match positions_angstrom without its final axis"
        )
    if positions.shape[-2] == 0:
        raise ValueError("at least one atom is required")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(charges)):
        raise ValueError("positions and charges must contain only finite values")

    if origin_angstrom is None:
        origin = np.zeros(3, dtype=np.float64)
        centered_positions = positions - origin
    else:
        origin = np.asarray(origin_angstrom, dtype=np.float64)
        if not np.all(np.isfinite(origin)):
            raise ValueError("origin_angstrom must contain only finite values")
        if origin.shape == (3,):
            centered_positions = positions - origin
        elif origin.shape == positions.shape[:-2] + (3,):
            centered_positions = positions - origin[..., None, :]
        else:
            raise ValueError(
                "origin_angstrom must have shape (3,) or (..., 3) matching "
                "the position batch dimensions"
            )

    if neutral_tolerance_e is not None:
        tolerance = float(neutral_tolerance_e)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("neutral_tolerance_e must be finite and non-negative")
        total_charge = np.sum(charges, axis=-1)
        if np.any(np.abs(total_charge) > tolerance):
            worst = float(np.max(np.abs(total_charge)))
            raise ValueError(
                "predicted charges are not neutral within the requested tolerance; "
                f"maximum |sum(q)| is {worst:.6g} e"
            )

    total = np.sum(charges[..., :, None] * centered_positions, axis=-2)
    if atomic_dipoles_e_angstrom is not None:
        atomic_dipoles = np.asarray(
            atomic_dipoles_e_angstrom,
            dtype=np.float64,
        )
        if atomic_dipoles.shape != positions.shape:
            raise ValueError(
                "atomic_dipoles_e_angstrom must have the same shape as positions"
            )
        if not np.all(np.isfinite(atomic_dipoles)):
            raise ValueError("atomic dipoles contain non-finite values")
        total = total + np.sum(atomic_dipoles, axis=-2)
    return _readonly(total)


def _flatten_force_samples(forces: np.ndarray, field: str) -> np.ndarray:
    values = np.asarray(forces, dtype=np.float64)
    if values.ndim == 3:
        n_displacements, n_atoms, dimensions = values.shape
        if dimensions != 3 or n_displacements != 3 * n_atoms:
            raise ValueError(
                f"{field} must have shape (3 * n_atoms, n_atoms, 3)"
            )
        values = values.reshape(n_displacements, 3 * n_atoms)
    elif values.ndim == 2:
        if (
            values.shape[0] == 0
            or values.shape[0] != values.shape[1]
            or values.shape[0] % 3
        ):
            raise ValueError(f"{field} must have shape (3N, 3N)")
    else:
        raise ValueError(
            f"{field} must have shape (3N, 3N) or (3N, N, 3)"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field} contains non-finite values")
    return values


def assemble_cartesian_hessian(
    forces_plus_eV_per_angstrom: np.ndarray,
    forces_minus_eV_per_angstrom: np.ndarray,
    step_angstrom: float,
) -> CartesianHessianEstimate:
    """Build ``H = -dF/dR`` from central differences of full-model forces.

    The first axis of each force array indexes the displaced coordinate.  The
    response forces are transposed into Hessian columns, so element ``H[i, j]``
    is ``-dF_i/dR_j``.  The raw matrix is retained for the antisymmetry check;
    the matrix intended for normal-mode analysis is its symmetric part.
    """

    step = _positive_step(step_angstrom)
    plus = _flatten_force_samples(
        forces_plus_eV_per_angstrom,
        "forces_plus_eV_per_angstrom",
    )
    minus = _flatten_force_samples(
        forces_minus_eV_per_angstrom,
        "forces_minus_eV_per_angstrom",
    )
    if plus.shape != minus.shape:
        raise ValueError("plus and minus force arrays must have the same shape")

    raw_eV_per_angstrom2 = -(plus - minus).T / (2.0 * step)
    symmetric_eV_per_angstrom2 = 0.5 * (
        raw_eV_per_angstrom2 + raw_eV_per_angstrom2.T
    )
    antisymmetric = raw_eV_per_angstrom2 - raw_eV_per_angstrom2.T
    max_abs_antisymmetry_eV = float(np.max(np.abs(antisymmetric), initial=0.0))
    scale = float(np.max(np.abs(symmetric_eV_per_angstrom2), initial=0.0))
    max_relative_antisymmetry = (
        max_abs_antisymmetry_eV / scale if scale > 0.0 else 0.0
    )
    raw_atomic = (
        raw_eV_per_angstrom2
        * EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    )
    symmetric_atomic = (
        symmetric_eV_per_angstrom2
        * EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    )

    return CartesianHessianEstimate(
        step_angstrom=step,
        raw_hessian_eV_per_angstrom2=_readonly(raw_eV_per_angstrom2),
        hessian_eV_per_angstrom2=_readonly(symmetric_eV_per_angstrom2),
        raw_hessian_hartree_per_bohr2=_readonly(raw_atomic),
        hessian_hartree_per_bohr2=_readonly(symmetric_atomic),
        max_abs_antisymmetry_eV_per_angstrom2=max_abs_antisymmetry_eV,
        max_abs_antisymmetry_hartree_per_bohr2=(
            max_abs_antisymmetry_eV
            * EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
        ),
        max_relative_antisymmetry=max_relative_antisymmetry,
    )


def assemble_dipole_derivative(
    dipoles_plus_e_angstrom: np.ndarray,
    dipoles_minus_e_angstrom: np.ndarray,
    step_angstrom: float,
) -> DipoleDerivativeEstimate:
    """Build ``d(mu)/dR`` using explicit atomic-unit conversions.

    Total dipoles in ``e * angstrom`` are multiplied by
    ``bohr / angstrom`` to obtain atomic dipole units (``e * bohr``).
    Likewise, the displacement is converted from angstrom to bohr.  The two
    length conversions cancel numerically in the derivative, but keeping both
    operations explicit prevents a unit mismatch when these arrays are
    compared with a quantum-chemistry dipole derivative.
    """

    step = _positive_step(step_angstrom)
    plus = np.asarray(dipoles_plus_e_angstrom, dtype=np.float64)
    minus = np.asarray(dipoles_minus_e_angstrom, dtype=np.float64)
    if (
        plus.ndim != 2
        or plus.shape[0] == 0
        or plus.shape[1] != 3
        or plus.shape[0] % 3
    ):
        raise ValueError("dipoles_plus_e_angstrom must have shape (3N, 3)")
    if plus.shape != minus.shape:
        raise ValueError("plus and minus dipole arrays must have the same shape")
    if not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
        raise ValueError("dipole arrays contain non-finite values")

    plus_atomic = plus * BOHR_PER_ANGSTROM
    minus_atomic = minus * BOHR_PER_ANGSTROM
    step_bohr = step * BOHR_PER_ANGSTROM
    derivative = (plus_atomic - minus_atomic) / (2.0 * step_bohr)
    return DipoleDerivativeEstimate(
        step_angstrom=step,
        dipoles_plus_e_angstrom=_readonly(plus),
        dipoles_minus_e_angstrom=_readonly(minus),
        dipoles_plus_atomic_units=_readonly(plus_atomic),
        dipoles_minus_atomic_units=_readonly(minus_atomic),
        dipole_derivative_3n_by_3_au=_readonly(derivative),
    )


def assemble_harmonic_ir_finite_difference(
    *,
    forces_plus_eV_per_angstrom: np.ndarray,
    forces_minus_eV_per_angstrom: np.ndarray,
    dipoles_plus_e_angstrom: np.ndarray,
    dipoles_minus_e_angstrom: np.ndarray,
    step_angstrom: float,
) -> HarmonicIRFiniteDifferenceEstimate:
    """Assemble the Hessian and dipole derivative for one displacement step."""

    hessian = assemble_cartesian_hessian(
        forces_plus_eV_per_angstrom,
        forces_minus_eV_per_angstrom,
        step_angstrom,
    )
    dipole = assemble_dipole_derivative(
        dipoles_plus_e_angstrom,
        dipoles_minus_e_angstrom,
        step_angstrom,
    )
    if hessian.hessian_hartree_per_bohr2.shape[0] != (
        dipole.dipole_derivative_3n_by_3_au.shape[0]
    ):
        raise ValueError("force and dipole samples describe different atom counts")
    return HarmonicIRFiniteDifferenceEstimate(
        hessian=hessian,
        dipole_derivative=dipole,
    )


def analyze_harmonic_ir(
    hessian_hartree_per_bohr2: np.ndarray,
    dipole_derivative_3n_by_3_au: np.ndarray,
    geometry_angstrom: np.ndarray,
    masses_u: np.ndarray,
) -> HarmonicIRModeAnalysis:
    """Project external motion and compute double-harmonic IR sticks.

    The Cartesian dipole derivative is ordered as ``(3N, 3)``: flattened
    atomic Cartesian coordinates by molecular dipole component.  For a
    mass-weighted eigenvector ``L`` the normal-coordinate derivative is

    ``d(mu)/dQ = (L / sqrt(mass_u)) @ d(mu)/dR``.

    Its squared Cartesian norm is converted to ``km / mol`` with the standard
    isotropic double-harmonic factor.  To analyze D2O, call this function again
    with deuterium masses while reusing the same geometry, Hessian, and dipole
    derivative.  Changing the potential or charge response would no longer be
    a mass-only isotope comparison.
    """

    geometry = np.asarray(geometry_angstrom, dtype=np.float64)
    masses = np.asarray(masses_u, dtype=np.float64)
    dipole_derivative = np.asarray(
        dipole_derivative_3n_by_3_au,
        dtype=np.float64,
    )
    if geometry.ndim != 2 or geometry.shape[1] != 3 or geometry.shape[0] < 2:
        raise ValueError(
            "geometry_angstrom must have shape (n_atoms, 3), n_atoms >= 2"
        )
    if not np.all(np.isfinite(geometry)):
        raise ValueError("geometry_angstrom contains non-finite values")
    if masses.shape != (geometry.shape[0],):
        raise ValueError("masses_u must contain one mass per atom")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("masses_u must contain only finite, positive values")
    if dipole_derivative.shape != (3 * geometry.shape[0], 3):
        raise ValueError("dipole_derivative_3n_by_3_au must have shape (3N, 3)")
    if not np.all(np.isfinite(dipole_derivative)):
        raise ValueError("dipole derivative contains non-finite values")

    solution = solve_vibrational_modes(
        hessian_hartree_per_bohr2,
        geometry,
        masses,
    )
    flattened_modes = solution.mass_weighted_modes.reshape(
        solution.frequencies_cm1.size,
        -1,
    )
    inverse_sqrt_mass = 1.0 / np.sqrt(np.repeat(masses, 3))
    normal_dipole_derivative = (
        flattened_modes * inverse_sqrt_mass[None, :]
    ) @ dipole_derivative
    intensities = IR_INTENSITY_KM_MOL_PER_E2_PER_U * np.einsum(
        "mi,mi->m",
        normal_dipole_derivative,
        normal_dipole_derivative,
    )

    return HarmonicIRModeAnalysis(
        masses_u=_readonly(masses),
        frequencies_cm1=_readonly(solution.frequencies_cm1),
        ir_intensities_km_mol=_readonly(intensities),
        normal_dipole_derivative_modes_by_3_e_per_sqrt_u=_readonly(
            normal_dipole_derivative
        ),
        mass_weighted_modes=_readonly(solution.mass_weighted_modes),
        external_rank=solution.external_rank,
    )


def harmonic_mode_dipole_strengths(
    dipole_derivative_3n_by_3_au: np.ndarray,
    mass_weighted_modes: np.ndarray,
    masses_u: np.ndarray,
) -> np.ndarray:
    """Return relative double-harmonic IR strengths for normal modes.

    ``mass_weighted_modes`` uses the convention returned by
    :func:`aux.reference.solve_vibrational_modes`.  Dividing each Cartesian
    component by ``sqrt(mass)`` converts a unit mass-weighted displacement to
    a Cartesian displacement.  The squared norm of the resulting mode dipole
    derivative is proportional to the integrated double-harmonic IR
    intensity.  No ``km / mol`` conversion is applied here.  Multiply by
    :data:`IR_INTENSITY_KM_MOL_PER_E2_PER_U` for absolute double-harmonic
    sticks, or use :func:`analyze_harmonic_ir` to obtain the complete result.
    """

    derivative = np.asarray(dipole_derivative_3n_by_3_au, dtype=np.float64)
    modes = np.asarray(mass_weighted_modes, dtype=np.float64)
    masses = np.asarray(masses_u, dtype=np.float64)
    if (
        masses.ndim != 1
        or masses.size == 0
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0.0)
    ):
        raise ValueError("masses_u must be a finite, non-empty positive vector")
    if derivative.shape != (3 * masses.size, 3):
        raise ValueError("dipole derivative must have shape (3N, 3)")
    if modes.ndim != 3 or modes.shape[1:] != (masses.size, 3):
        raise ValueError("mass_weighted_modes must have shape (n_modes, N, 3)")
    if not np.all(np.isfinite(derivative)) or not np.all(np.isfinite(modes)):
        raise ValueError("dipole derivative and modes must be finite")

    inverse_sqrt_mass = 1.0 / np.sqrt(np.repeat(masses, 3))
    cartesian_modes = modes.reshape(modes.shape[0], -1) * inverse_sqrt_mass
    mode_dipole_derivatives = cartesian_modes @ derivative
    strengths = np.sum(mode_dipole_derivatives**2, axis=1)
    return _readonly(np.maximum(strengths, 0.0))


def _max_relative_change(current: np.ndarray, reference: np.ndarray) -> float:
    maximum_change = float(np.max(np.abs(current - reference), initial=0.0))
    scale = float(np.max(np.abs(reference), initial=0.0))
    if scale > 0.0:
        return maximum_change / scale
    return 0.0 if maximum_change == 0.0 else float("inf")


def summarize_finite_difference_convergence(
    estimates: Sequence[HarmonicIRFiniteDifferenceEstimate],
) -> FiniteDifferenceConvergence:
    """Compare every displacement step with the next finer available step."""

    ordered = tuple(
        sorted(estimates, key=lambda item: item.step_angstrom, reverse=True)
    )
    if len(ordered) < 2:
        raise ValueError("at least two finite-difference estimates are required")
    steps = np.asarray([item.step_angstrom for item in ordered], dtype=np.float64)
    if np.unique(steps).size != steps.size:
        raise ValueError("finite-difference steps must be unique")

    hessian_shape = ordered[0].hessian_hartree_per_bohr2.shape
    dipole_shape = ordered[0].dipole_derivative_3n_by_3_au.shape
    for estimate in ordered[1:]:
        if estimate.hessian_hartree_per_bohr2.shape != hessian_shape:
            raise ValueError("all Hessian estimates must have the same shape")
        if estimate.dipole_derivative_3n_by_3_au.shape != dipole_shape:
            raise ValueError("all dipole-derivative estimates must have the same shape")

    hessian_absolute: list[float] = []
    hessian_relative: list[float] = []
    dipole_absolute: list[float] = []
    dipole_relative: list[float] = []
    for coarse, fine in zip(ordered[:-1], ordered[1:], strict=True):
        coarse_hessian = coarse.hessian_hartree_per_bohr2
        fine_hessian = fine.hessian_hartree_per_bohr2
        coarse_dipole = coarse.dipole_derivative_3n_by_3_au
        fine_dipole = fine.dipole_derivative_3n_by_3_au
        hessian_absolute.append(
            float(np.max(np.abs(coarse_hessian - fine_hessian), initial=0.0))
        )
        hessian_relative.append(_max_relative_change(coarse_hessian, fine_hessian))
        dipole_absolute.append(
            float(np.max(np.abs(coarse_dipole - fine_dipole), initial=0.0))
        )
        dipole_relative.append(_max_relative_change(coarse_dipole, fine_dipole))

    return FiniteDifferenceConvergence(
        steps_angstrom=_readonly(steps),
        hessian_max_relative_antisymmetry=_readonly(
            np.asarray(
                [item.hessian.max_relative_antisymmetry for item in ordered]
            )
        ),
        coarse_steps_angstrom=_readonly(steps[:-1]),
        fine_steps_angstrom=_readonly(steps[1:]),
        hessian_max_abs_change_hartree_per_bohr2=_readonly(
            np.asarray(hessian_absolute)
        ),
        hessian_max_relative_change=_readonly(np.asarray(hessian_relative)),
        dipole_derivative_max_abs_change_au=_readonly(
            np.asarray(dipole_absolute)
        ),
        dipole_derivative_max_relative_change=_readonly(
            np.asarray(dipole_relative)
        ),
    )


def summarize_harmonic_ir_convergence(
    estimates: Sequence[HarmonicIRFiniteDifferenceEstimate],
    geometry_angstrom: np.ndarray,
    masses_u: np.ndarray,
) -> HarmonicIRConvergence:
    """Track frequencies, intensities, and mode continuity across FD steps.

    The same geometry and mass vector are used at every step.  This reports
    adjacent-step changes rather than silently accepting the smallest step.
    For the tutorial, frequency stability and weak-mode intensity stability
    should both be inspected; a stable Hessian alone does not prove a stable
    spectrum.
    """

    ordered = tuple(
        sorted(estimates, key=lambda item: item.step_angstrom, reverse=True)
    )
    finite_difference = summarize_finite_difference_convergence(ordered)
    analyses = tuple(
        analyze_harmonic_ir(
            estimate.hessian_hartree_per_bohr2,
            estimate.dipole_derivative_3n_by_3_au,
            geometry_angstrom,
            masses_u,
        )
        for estimate in ordered
    )
    frequencies = np.stack(
        [analysis.frequencies_cm1 for analysis in analyses],
        axis=0,
    )
    intensities = np.stack(
        [analysis.ir_intensities_km_mol for analysis in analyses],
        axis=0,
    )
    frequency_change = np.abs(frequencies[:-1] - frequencies[1:])
    intensity_change = np.abs(intensities[:-1] - intensities[1:])
    intensity_scale = np.abs(intensities[1:])
    intensity_relative = np.full_like(intensity_change, np.inf)
    np.divide(
        intensity_change,
        intensity_scale,
        out=intensity_relative,
        where=intensity_scale > 0.0,
    )
    intensity_relative[
        (intensity_change == 0.0) & (intensity_scale == 0.0)
    ] = 0.0

    flattened_modes = np.stack(
        [
            analysis.mass_weighted_modes.reshape(frequencies.shape[1], -1)
            for analysis in analyses
        ],
        axis=0,
    )
    overlaps = np.abs(
        np.sum(flattened_modes[:-1] * flattened_modes[1:], axis=2)
    ) ** 2

    return HarmonicIRConvergence(
        finite_difference=finite_difference,
        steps_angstrom=_readonly(finite_difference.steps_angstrom),
        frequencies_cm1=_readonly(frequencies),
        ir_intensities_km_mol=_readonly(intensities),
        frequency_abs_change_cm1=_readonly(frequency_change),
        frequency_max_abs_change_cm1=_readonly(
            np.max(frequency_change, axis=1)
        ),
        ir_intensity_abs_change_km_mol=_readonly(intensity_change),
        ir_intensity_relative_change=_readonly(intensity_relative),
        ir_intensity_max_abs_change_km_mol=_readonly(
            np.max(intensity_change, axis=1)
        ),
        ir_intensity_max_relative_change=_readonly(
            np.max(intensity_relative, axis=1)
        ),
        same_index_mode_squared_overlaps=_readonly(overlaps),
        minimum_same_index_mode_squared_overlap=_readonly(
            np.min(overlaps, axis=1)
        ),
    )


__all__ = [
    "ANGSTROM_PER_BOHR",
    "BOHR_PER_ANGSTROM",
    "EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2",
    "EV_PER_HARTREE",
    "HARTREE_PER_EV",
    "IR_INTENSITY_KM_MOL_PER_E2_PER_U",
    "CartesianDisplacements",
    "CartesianHessianEstimate",
    "DipoleDerivativeEstimate",
    "FiniteDifferenceConvergence",
    "HarmonicIRConvergence",
    "HarmonicIRFiniteDifferenceEstimate",
    "HarmonicIRModeAnalysis",
    "analyze_harmonic_ir",
    "assemble_cartesian_hessian",
    "assemble_dipole_derivative",
    "assemble_harmonic_ir_finite_difference",
    "harmonic_mode_dipole_strengths",
    "molecular_dipoles_from_atomic_predictions",
    "summarize_finite_difference_convergence",
    "summarize_harmonic_ir_convergence",
    "symmetric_cartesian_displacements",
]
