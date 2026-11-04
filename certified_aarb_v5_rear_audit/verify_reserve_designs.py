"""Independent consistency audit for the 1%, 2% and 5% rear-reserve designs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
COMPARISON = OUT / "comparison"
TARGETS = ((1, 0.01), (2, 0.02), (5, 0.05))
TOL = 2e-10


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def point_key(row):
    return (
        row["set"],
        round(float(row["speed_kmh"]), 12),
        round(float(row["mass_fraction"]), 12),
        round(float(row["payload_height"]), 12),
    )


def close(left, right, atol=5e-13):
    return abs(float(left) - float(right)) <= atol


def main():
    bank = load(ROOT / "certified_aarb_v4/outputs/envelope_results.json")
    expected_points = {point_key(row) for row in bank["rows"]}
    if len(expected_points) != 170:
        raise RuntimeError("archived operating bank does not contain 170 unique points")

    records = []
    overall_checks = []
    for percent, target in TARGETS:
        directory = OUT / f"reserve_{percent}pct"
        summary = load(directory / "summary.json")
        design = load(directory / "design.json")
        screening = load(directory / "screening.json")
        rear = load(directory / "rear_verification.json")
        coefficients = np.loadtxt(directory / "coefficients.csv", delimiter=",")
        screening_points = {point_key(row) for row in screening["rows"]}
        rear_points = {point_key(row) for row in rear["rows"]}

        proposed = [row["proposed"] for row in screening["rows"]]
        relative = np.asarray([row["relative_margin"] for row in rear["rows"]])
        absolute = np.asarray([row["certified_margin"] for row in rear["rows"]])
        recomputed = {
            "worst_front": max(row["front_LTR"] for row in proposed),
            "max_constraint_utilization": max(
                row["constraint_utilization"] for row in proposed
            ),
            "max_spectral_radius": max(row["spectral_radius"] for row in proposed),
            "minimum_relative_margin": float(relative.min()),
            "minimum_absolute_margin": float(absolute.min()),
        }

        source_hash_checks = {}
        source_map = {
            "run_reserve_design.py": HERE / "run_reserve_design.py",
            "v4_study.py": ROOT / "certified_aarb_v4/study.py",
            "v3_study.py": ROOT / "certified_aarb_v3/study.py",
            "vehicle_model.py": ROOT / "validation/vehicle_model.py",
            "parameters.py": ROOT / "validation/parameters.py",
        }
        for name, path in source_map.items():
            source_hash_checks[name] = design["source_hashes"].get(name) == sha256(path)

        checks = {
            "summary_pass": summary["status"] == "PASS",
            "target_matches_protocol": close(summary["target_relative_margin"], target),
            "coefficient_shape_2_by_6": coefficients.shape == (2, 6),
            "coefficients_finite": bool(np.isfinite(coefficients).all()),
            "coefficients_match_design_record": bool(np.allclose(
                coefficients.reshape(-1),
                np.asarray(design["coefficients"]).reshape(-1),
                rtol=0.0,
                atol=5e-13,
            )),
            "screening_has_170_points": screening["count"] == len(screening["rows"]) == 170,
            "rear_audit_has_170_points": rear["count"] == len(rear["rows"]) == 170,
            "screening_bank_matches_archive": screening_points == expected_points,
            "rear_bank_matches_archive": rear_points == expected_points,
            "all_rear_margins_nonnegative": bool(np.all(absolute >= -TOL)),
            "requested_reserve_achieved": bool(np.all(relative >= target - TOL)),
            "all_constraints_feasible": recomputed["max_constraint_utilization"] <= 1.0 + TOL,
            "all_closed_loops_stable": recomputed["max_spectral_radius"] < 1.0,
            "summary_worst_front_consistent": close(
                summary["worst_front_LTR_21_phase"], recomputed["worst_front"]
            ),
            "summary_constraint_consistent": close(
                summary["max_constraint_utilization_21_phase"],
                recomputed["max_constraint_utilization"],
            ),
            "summary_spectral_radius_consistent": close(
                summary["max_spectral_radius"], recomputed["max_spectral_radius"]
            ),
            "summary_minimum_relative_margin_consistent": close(
                summary["minimum_relative_rear_margin"],
                recomputed["minimum_relative_margin"],
            ),
            "summary_minimum_absolute_margin_consistent": close(
                summary["minimum_absolute_rear_margin_81_phase"],
                recomputed["minimum_absolute_margin"],
            ),
            "source_hashes_match": all(source_hash_checks.values()),
        }
        passed = all(checks.values())
        overall_checks.append(passed)
        records.append({
            "target_percent": percent,
            "status": "PASS" if passed else "FAIL",
            "checks": checks,
            "source_hash_checks": source_hash_checks,
            "recomputed": recomputed,
            "coefficient_sha256": sha256(directory / "coefficients.csv"),
        })

    comparison = load(COMPARISON / "reserve_comparison.json")
    selected = comparison["selection"]
    five = next(row for row in comparison["designs"] if row["key"] == "reserve_5pct")
    selection_checks = {
        "comparison_pass": comparison["status"] == "PASS",
        "five_percent_selected": selected["key"] == "reserve_5pct",
        "largest_declared_reserve_selected": five["achieved_minimum_margin_percent"] == max(
            row["achieved_minimum_margin_percent"] for row in comparison["designs"]
        ),
        "front_cost_below_half_percent": five["front_cost_vs_current_percent"] < 0.5,
        "comparison_constraints_feasible": comparison["key_findings"]["all_constraints_feasible"],
        "comparison_loops_stable": comparison["key_findings"]["all_closed_loops_stable"],
    }
    status = "PASS" if all(overall_checks) and all(selection_checks.values()) else "FAIL"
    result = {
        "status": status,
        "date": "2026-09-23",
        "scope": "Final audit of reserve designs on the archived 170-plant bank.",
        "point_count": 170,
        "designs": records,
        "selection_checks": selection_checks,
        "selected_controller": selected,
        "limitations": [
            "The audit is pointwise on the evaluated bank.",
            "It is not a continuous speed-payload or hardware-uncertainty guarantee.",
        ],
    }
    (COMPARISON / "final_reserve_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    manifest = {}
    for path in sorted(
        item for item in COMPARISON.rglob("*")
        if item.is_file() and item.name != "artifact_manifest.json"
    ):
        manifest[str(path.relative_to(HERE))] = sha256(path)
    (COMPARISON / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": status,
        "selected": selected["label"],
        "achieved_margin_percent": selected["achieved_minimum_margin_percent"],
        "front_cost_percent": selected["front_cost_vs_current_percent"],
    }, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()