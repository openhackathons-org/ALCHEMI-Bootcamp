#!/usr/bin/env python3
"""Model-free grader for the SEI Pareto challenge."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Optional


REQUIRED_COLUMNS = {
    "candidate_id",
    "role",
    "molecule_class",
    "passivating_surface_id",
    "E_bind_Li_eV",
    "E_bind_passivating_eV",
    "seeding_score",
    "passivation_score",
    "is_pareto",
    "hypervolume_improvement",
    "selected",
}
RAW_ENERGY_COLUMNS = {
    "candidate_id",
    "interaction",
    "surface_id",
    "E_surface_species_eV",
    "E_surface_eV",
    "E_species_eV",
}
FLOAT_TOL = 1e-6
ENERGY_TOL = 1e-4


class GradeError(ValueError):
    """Raised for a submission that does not satisfy the challenge contract."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise GradeError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise GradeError(f"Cannot parse boolean value: {value!r}")


def _float(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        ident = row.get("candidate_id", "<unknown>")
        raise GradeError(f"Invalid numeric value for {column!r} in {ident!r}") from exc


# Reward windows -- kept identical to challenge_utils/rewards.py (the scoring
# source of truth); duplicated so this grader stays standalone.
SEEDING_WEAK_EDGE_EV = 1.00
SEEDING_IDEAL_LOW_EV = 1.40
SEEDING_IDEAL_HIGH_EV = 1.80
SEEDING_STRONG_EDGE_EV = 3.00
PASSIVATION_FULL_REWARD_EV = 0.60
PASSIVATION_ZERO_REWARD_EV = 1.30


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _binding_strength(e_bind: float) -> float:
    """Exothermic adsorption-strength magnitude: strength = max(0, -E_bind)."""
    return max(0.0, -float(e_bind))


def _score_from_li_binding(e_bind: float) -> float:
    """Seeding score: Sabatier window on adsorption strength (see rewards.py).

    Full reward for strengths 0.8-1.5 eV, linear tapers to zero below 0.5 eV
    and above 2.0 eV.
    """
    strength = _binding_strength(e_bind)
    if strength <= SEEDING_WEAK_EDGE_EV:
        return 0.0
    if strength < SEEDING_IDEAL_LOW_EV:
        return _clip01(
            (strength - SEEDING_WEAK_EDGE_EV)
            / (SEEDING_IDEAL_LOW_EV - SEEDING_WEAK_EDGE_EV)
        )
    if strength <= SEEDING_IDEAL_HIGH_EV:
        return 1.0
    if strength < SEEDING_STRONG_EDGE_EV:
        return _clip01(
            (SEEDING_STRONG_EDGE_EV - strength)
            / (SEEDING_STRONG_EDGE_EV - SEEDING_IDEAL_HIGH_EV)
        )
    return 0.0


def _score_from_passivating_binding(e_bind: float) -> float:
    """Passivation score: weak-adsorption reward on strength (see rewards.py).

    Full reward for strength <= 0.3 eV, linear taper to zero by 0.8 eV.
    """
    strength = _binding_strength(e_bind)
    if strength <= PASSIVATION_FULL_REWARD_EV:
        return 1.0
    if strength >= PASSIVATION_ZERO_REWARD_EV:
        return 0.0
    return _clip01(
        1.0
        - (strength - PASSIVATION_FULL_REWARD_EV)
        / (PASSIVATION_ZERO_REWARD_EV - PASSIVATION_FULL_REWARD_EV)
    )


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] >= b[0] and a[1] >= b[1] and (a[0] > b[0] or a[1] > b[1])


def pareto_flags(points: list[tuple[float, float]]) -> list[bool]:
    flags: list[bool] = []
    for i, point in enumerate(points):
        flags.append(not any(_dominates(other, point) for j, other in enumerate(points) if i != j))
    return flags


def hypervolume_2d(points: list[tuple[float, float]]) -> float:
    """Return dominated area for 2D maximization against reference point (0, 0)."""
    if not points:
        return 0.0
    clipped = [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for x, y in points]
    front = [p for p, keep in zip(clipped, pareto_flags(clipped)) if keep]
    front.sort(key=lambda p: (p[0], -p[1]))
    area = 0.0
    previous_x = 0.0
    for x, y in front:
        if x > previous_x:
            area += (x - previous_x) * y
            previous_x = x
    return area


def _validate_required_columns(rows: list[dict[str, str]], required: set[str]) -> None:
    if not rows:
        raise GradeError("Submission CSV has no rows.")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise GradeError(f"Missing required column(s): {', '.join(missing)}")


def _validate_raw_energies(rows: list[dict[str, str]], raw_path: Path) -> None:
    if not raw_path.exists():
        return
    raw_rows = _read_csv(raw_path)
    _validate_required_columns(raw_rows, RAW_ENERGY_COLUMNS)
    raw_by_key = {
        (row["candidate_id"], row["interaction"]): row
        for row in raw_rows
    }
    for row in rows:
        candidate_id = row["candidate_id"]
        for interaction, output_column in (
            ("li_metal", "E_bind_Li_eV"),
            ("passivating", "E_bind_passivating_eV"),
        ):
            raw = raw_by_key.get((candidate_id, interaction))
            if raw is None:
                raise GradeError(
                    f"Raw energy check enabled, but missing {interaction!r} row for {candidate_id!r}"
                )
            expected = (
                float(raw["E_surface_species_eV"])
                - float(raw["E_surface_eV"])
                - float(raw["E_species_eV"])
            )
            observed = _float(row, output_column)
            if abs(observed - expected) > ENERGY_TOL:
                raise GradeError(
                    f"{candidate_id}: {output_column}={observed:.8f} does not match "
                    f"raw component energies ({expected:.8f})"
                )


def grade_submission(path: Path, raw_path: Optional[Path] = None) -> dict[str, object]:
    rows = _read_csv(path)
    _validate_required_columns(rows, REQUIRED_COLUMNS)
    raw_path = raw_path or path.parent / "raw_component_energies.csv"

    points: list[tuple[float, float]] = []
    baseline_points: list[tuple[float, float]] = []
    selected: list[str] = []
    additive_improvements: dict[str, float] = {}

    for row in rows:
        candidate_id = row["candidate_id"]
        seeding = _float(row, "seeding_score")
        passivation = _float(row, "passivation_score")
        if not (0.0 - FLOAT_TOL <= seeding <= 1.0 + FLOAT_TOL):
            raise GradeError(f"{candidate_id}: seeding_score must be in [0, 1]")
        if not (0.0 - FLOAT_TOL <= passivation <= 1.0 + FLOAT_TOL):
            raise GradeError(f"{candidate_id}: passivation_score must be in [0, 1]")

        expected_seed = _score_from_li_binding(_float(row, "E_bind_Li_eV"))
        expected_pass = _score_from_passivating_binding(_float(row, "E_bind_passivating_eV"))
        if abs(seeding - expected_seed) > FLOAT_TOL:
            raise GradeError(f"{candidate_id}: seeding_score does not match challenge formula")
        if abs(passivation - expected_pass) > FLOAT_TOL:
            raise GradeError(f"{candidate_id}: passivation_score does not match challenge formula")

        point = (seeding, passivation)
        points.append(point)
        if row["role"].strip().lower() == "baseline":
            baseline_points.append(point)
        if _bool(row["selected"]):
            selected.append(candidate_id)

    if len(baseline_points) < 2:
        raise GradeError("At least two baseline rows are required.")

    expected_pareto = pareto_flags(points)
    for row, expected in zip(rows, expected_pareto):
        observed = _bool(row["is_pareto"])
        if observed != expected:
            raise GradeError(f"{row['candidate_id']}: is_pareto should be {expected}")

    baseline_hv = hypervolume_2d(baseline_points)
    for row, point in zip(rows, points):
        if row["role"].strip().lower() == "baseline":
            expected_improvement = 0.0
        else:
            expected_improvement = hypervolume_2d([*baseline_points, point]) - baseline_hv
            additive_improvements[row["candidate_id"]] = expected_improvement
        observed = _float(row, "hypervolume_improvement")
        if abs(observed - expected_improvement) > FLOAT_TOL:
            raise GradeError(
                f"{row['candidate_id']}: hypervolume_improvement should be "
                f"{expected_improvement:.8f}, got {observed:.8f}"
            )

    if not selected:
        raise GradeError("Exactly one additive must be marked selected; found none.")
    if len(selected) != 1:
        raise GradeError(
            f"Exactly one additive must be marked selected; found {len(selected)}."
        )
    selected_roles = {
        row["candidate_id"]: row["role"].strip().lower()
        for row in rows
    }
    if any(selected_roles.get(candidate_id) == "baseline" for candidate_id in selected):
        raise GradeError("Baseline rows cannot be selected as additives.")

    best = max(additive_improvements.values()) if additive_improvements else 0.0
    acceptable = {
        candidate_id
        for candidate_id, improvement in additive_improvements.items()
        if abs(improvement - best) <= FLOAT_TOL
    }
    bad_selected = [candidate_id for candidate_id in selected if candidate_id not in acceptable]
    if bad_selected:
        raise GradeError(
            "Selected additive does not maximize hypervolume improvement. "
            f"Selected={bad_selected}; acceptable={sorted(acceptable)}"
        )

    _validate_raw_energies(rows, raw_path)
    return {
        "status": "pass",
        "n_rows": len(rows),
        "selected": sorted(selected),
        "best_hypervolume_improvement": best,
        "raw_energy_check": raw_path.exists(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission",
        nargs="?",
        type=Path,
        default=Path("outputs/challenge_submission.csv"),
        help="Path to challenge_submission.csv",
    )
    parser.add_argument(
        "--raw-energies",
        type=Path,
        default=None,
        help="Optional raw_component_energies.csv path",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = grade_submission(args.submission, raw_path=args.raw_energies)
    except GradeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS: selected="
        + ",".join(result["selected"])
        + f"; best_hypervolume_improvement={result['best_hypervolume_improvement']:.6f}; "
        + f"raw_energy_check={result['raw_energy_check']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
