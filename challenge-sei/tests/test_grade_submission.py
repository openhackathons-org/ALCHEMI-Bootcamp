from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GRADER_PATH = ROOT / "scripts" / "grade_submission.py"


def _load_grader():
    spec = importlib.util.spec_from_file_location("grade_submission", GRADER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _base_rows(selected: str = "FEC", *, tie: bool = False) -> list[dict[str, object]]:
    grader = _load_grader()
    # E_bind that place scores in the reward windows (rewards.py) with a clear
    # hypervolume winner (FEC); VC a lesser additive, TMP dominated.
    rows = [
        ("EC", "baseline", "carbonate", "Li2CO3", -2.10, -1.125),
        ("EMC", "baseline", "carbonate", "Li2CO3", -2.70, -0.775),
        ("FEC", "additive", "fluorinated", "LiF", -1.60, -0.50),
        ("VC", "additive", "carbonate", "Li2CO3", -1.20, -0.95),
        ("TMP", "additive", "phosphate", "Li3PO4", -2.70, -1.125),
    ]
    if tie:
        rows[3] = ("VC", "additive", "carbonate", "Li2CO3", -1.60, -0.50)

    points = []
    for _, _, _, _, e_li, e_pass in rows:
        points.append((
            grader._score_from_li_binding(e_li),
            grader._score_from_passivating_binding(e_pass),
        ))
    baseline = [point for point, row in zip(points, rows) if row[1] == "baseline"]
    baseline_hv = grader.hypervolume_2d(baseline)
    pareto = grader.pareto_flags(points)

    output = []
    for row, point, is_pareto in zip(rows, points, pareto):
        candidate_id, role, molecule_class, surface, e_li, e_pass = row
        improvement = 0.0 if role == "baseline" else grader.hypervolume_2d([*baseline, point]) - baseline_hv
        output.append({
            "candidate_id": candidate_id,
            "role": role,
            "molecule_class": molecule_class,
            "passivating_surface_id": surface,
            "E_bind_Li_eV": e_li,
            "E_bind_passivating_eV": e_pass,
            "seeding_score": point[0],
            "passivation_score": point[1],
            "is_pareto": is_pareto,
            "hypervolume_improvement": improvement,
            "selected": candidate_id == selected,
        })
    return output


def _load_rewards():
    spec = importlib.util.spec_from_file_location(
        "rewards", ROOT / "challenge_utils" / "rewards.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grader_scoring_matches_rewards_module():
    """The grader's inline reward formulas must stay identical to
    challenge_utils/rewards.py (the scoring source of truth)."""
    grader = _load_grader()
    rewards = _load_rewards()
    # sweep -3.0 .. +1.0 eV, dense enough to hit every window edge and taper
    for i in range(-300, 101):
        e_bind = i / 100.0
        assert grader._score_from_li_binding(e_bind) == pytest.approx(
            rewards.seeding_score(e_bind), abs=1e-12
        ), f"seeding mismatch at E_bind={e_bind}"
        assert grader._score_from_passivating_binding(e_bind) == pytest.approx(
            rewards.passivation_score(e_bind), abs=1e-12
        ), f"passivation mismatch at E_bind={e_bind}"


def test_valid_submission_with_clear_hypervolume_winner_passes(tmp_path: Path):
    grader = _load_grader()
    submission = _write_csv(tmp_path / "challenge_submission.csv", _base_rows())

    result = grader.grade_submission(submission)

    assert result["status"] == "pass"
    assert result["selected"] == ["FEC"]
    assert result["raw_energy_check"] is False


def test_wrong_selected_additive_fails(tmp_path: Path):
    grader = _load_grader()
    submission = _write_csv(tmp_path / "challenge_submission.csv", _base_rows(selected="TMP"))

    with pytest.raises(grader.GradeError, match="does not maximize"):
        grader.grade_submission(submission)


def test_missing_required_column_fails(tmp_path: Path):
    grader = _load_grader()
    rows = _base_rows()
    for row in rows:
        row.pop("is_pareto")
    submission = _write_csv(tmp_path / "challenge_submission.csv", rows)

    with pytest.raises(grader.GradeError, match="Missing required column"):
        grader.grade_submission(submission)


def test_tied_maximum_hypervolume_accepts_any_tied_selected_additive(tmp_path: Path):
    grader = _load_grader()
    submission = _write_csv(
        tmp_path / "challenge_submission.csv",
        _base_rows(selected="VC", tie=True),
    )

    result = grader.grade_submission(submission)

    assert result["status"] == "pass"
    assert result["selected"] == ["VC"]


def test_multiple_selected_rows_fail(tmp_path: Path):
    grader = _load_grader()
    rows = _base_rows()
    for row in rows:
        if row["candidate_id"] in {"FEC", "VC"}:
            row["selected"] = True
    submission = _write_csv(tmp_path / "challenge_submission.csv", rows)

    with pytest.raises(grader.GradeError, match="Exactly one additive"):
        grader.grade_submission(submission)


def test_optional_raw_component_energies_check_binding_formula(tmp_path: Path):
    grader = _load_grader()
    rows = _base_rows()
    submission = _write_csv(tmp_path / "challenge_submission.csv", rows)

    raw_rows = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        e_species = -10.0
        for interaction, surface, bind_column, e_surface in (
            ("li_metal", "Li_metal", "E_bind_Li_eV", -100.0),
            ("passivating", row["passivating_surface_id"], "E_bind_passivating_eV", -80.0),
        ):
            raw_rows.append({
                "candidate_id": candidate_id,
                "interaction": interaction,
                "surface_id": surface,
                "E_surface_species_eV": e_surface + e_species + float(row[bind_column]),
                "E_surface_eV": e_surface,
                "E_species_eV": e_species,
            })
    raw = _write_csv(tmp_path / "raw_component_energies.csv", raw_rows)

    result = grader.grade_submission(submission, raw_path=raw)

    assert result["status"] == "pass"
    assert result["raw_energy_check"] is True
