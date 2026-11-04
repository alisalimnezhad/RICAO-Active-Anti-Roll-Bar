"""Checkpointed rear-reserve design and 170-point verification.

One target is processed per invocation so an interrupted Prism session can be
resumed without repeating completed design or verification work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V3 = ROOT / "certified_aarb_v3"
V4 = ROOT / "certified_aarb_v4"
if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))

from study import (  # noqa: E402
    ParametricStudy,
    RobustDesigner,
    base,
    base_caps,
    compact_evaluate,
    design_skeleton,
    study_at,
)
from run_envelope import controller_summary, points  # noqa: E402
from verify_envelope import rear_pair_certificate  # noqa: E402


PHASES_SCREEN = 21
PHASES_REAR = 81


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_stem(target):
    return f"reserve_{100.0 * float(target):g}pct".replace(".", "p")


def load_rows(path):
    return json.loads(path.read_text()) if path.exists() else []


def point_key(row):
    return (
        round(float(row["speed_kmh"]), 12),
        round(float(row["mass_fraction"]), 12),
        round(float(row["payload_height"]), 12),
        row.get("set", ""),
    )


def synthesize(target, guard, out):
    design_path = out / "design.json"
    coefficient_path = out / "coefficients.csv"
    if design_path.exists() and coefficient_path.exists():
        print("checkpoint: loading completed design", flush=True)
        return json.loads(design_path.read_text()), np.loadtxt(
            coefficient_path, delimiter=","
        ).reshape(-1)

    start = time.time()
    nominal = ParametricStudy(speed_kmh=70.0)
    fixed_gain = nominal.K.copy()
    studies = design_skeleton(fixed_gain, phases=9)
    zero = np.zeros(12)
    current = np.loadtxt(
        V4 / "outputs" / "fixed_rear_protected_coefficients.csv",
        delimiter=",",
    ).reshape(-1)
    old_protected = np.loadtxt(
        V3 / "outputs" / "bank_fixed_rear_protected_coefficients.csv",
        delimiter=",",
    ).reshape(-1)
    old_balanced = np.loadtxt(
        V3 / "outputs" / "bank_fixed_balanced_coefficients.csv",
        delimiter=",",
    ).reshape(-1)

    baselines = []
    caps = base_caps(studies, current_caps=[0.985, 0.985], other_margin=0.980)
    for index, study in enumerate(studies):
        evaluation = study.evaluate(zero, phases=41)
        baseline = {
            "label": study.label,
            "rear_grid_upper": float(evaluation["grid_upper"][1]),
            "rear_continuous_upper": float(evaluation["continuous_upper"][1]),
            "rear_intersample_correction": float(
                evaluation["interpolation_correction"][1]
            ),
        }
        baselines.append(baseline)
        caps[index, 1] = baseline["rear_grid_upper"] - float(guard)

    designer = RobustDesigner(studies, objective_outputs=(0,), caps=caps)
    result = designer.synthesize(
        [zero, current, old_protected, old_balanced],
        tolerance=7e-6,
        verbose=True,
    )

    design_checks = []
    for study in studies:
        evaluation = study.evaluate(result.f, phases=PHASES_REAR)
        design_checks.append({
            "label": study.label,
            "front_continuous_upper": float(evaluation["continuous_upper"][0]),
            "rear_continuous_upper": float(evaluation["continuous_upper"][1]),
            "max_non_ltr_utilization": float(
                np.max(evaluation["continuous_upper"][2:])
            ),
            "spectral_radius": float(study.rho_closed),
        })

    record = {
        "date": "2026-09-21",
        "target_relative_margin": float(target),
        "design_absolute_grid_guard": float(guard),
        "coefficients": result.f,
        "design_gamma": result.gamma,
        "scenario_lower": result.lower,
        "iterations": len(result.history),
        "scenario_count": len(result.scenarios),
        "history": result.history,
        "baseline_design_models": baselines,
        "design_model_checks_81_phase": design_checks,
        "runtime_seconds": time.time() - start,
        "source_hashes": {
            "run_reserve_design.py": sha256(Path(__file__)),
            "v4_study.py": sha256(V4 / "study.py"),
            "v3_study.py": sha256(V3 / "study.py"),
            "vehicle_model.py": sha256(ROOT / "validation" / "vehicle_model.py"),
            "parameters.py": sha256(ROOT / "validation" / "parameters.py"),
        },
    }
    base.save_json(design_path, record)
    np.savetxt(coefficient_path, result.f.reshape(2, 6), delimiter=",")
    print("checkpoint: design saved", flush=True)
    return record, result.f


def broad_screen(coefficients, out):
    final_path = out / "screening.json"
    if final_path.exists():
        print("checkpoint: loading completed 21-phase screening", flush=True)
        return json.loads(final_path.read_text())

    partial_path = out / "screening_partial.json"
    rows = load_rows(partial_path)
    completed = {point_key(row) for row in rows}
    nominal = ParametricStudy(speed_kmh=70.0)
    fixed_gain = nominal.K.copy()
    all_points = points()
    start = time.time()

    for index, point in enumerate(all_points, 1):
        if point_key(point) in completed:
            continue
        study = study_at(
            point["speed_kmh"],
            point["mass_fraction"],
            point["payload_height"],
            fixed_gain,
            phases=9,
        )
        evaluation = compact_evaluate(
            study, coefficients, phases=PHASES_SCREEN
        )
        row = dict(point)
        row["proposed"] = controller_summary(study, evaluation)
        rows.append(row)
        completed.add(point_key(point))
        if len(rows) % 5 == 0:
            base.save_json(partial_path, rows)
            print(f"screening checkpoint {len(rows)}/{len(all_points)}", flush=True)

    base.save_json(partial_path, rows)
    worst_front = max(rows, key=lambda row: row["proposed"]["front_LTR"])
    worst_ltr = max(rows, key=lambda row: row["proposed"]["worst_LTR"])
    worst_util = max(
        rows, key=lambda row: row["proposed"]["constraint_utilization"]
    )
    worst_rho = max(rows, key=lambda row: row["proposed"]["spectral_radius"])
    record = {
        "phase_count": PHASES_SCREEN,
        "count": len(rows),
        "worst_front_LTR": worst_front["proposed"]["front_LTR"],
        "worst_front_point": {
            key: worst_front[key] for key in
            ("set", "speed_kmh", "mass_fraction", "payload_height")
        },
        "worst_LTR": worst_ltr["proposed"]["worst_LTR"],
        "worst_LTR_point": {
            key: worst_ltr[key] for key in
            ("set", "speed_kmh", "mass_fraction", "payload_height")
        },
        "max_constraint_utilization": worst_util["proposed"]["constraint_utilization"],
        "limiting_constraint": worst_util["proposed"]["limiting_constraint"],
        "max_spectral_radius": worst_rho["proposed"]["spectral_radius"],
        "rows": rows,
        "runtime_seconds_current_invocation": time.time() - start,
    }
    base.save_json(final_path, record)
    print("checkpoint: screening saved", flush=True)
    return record


def fine_rear_audit(coefficients, target, out):
    final_path = out / "rear_verification.json"
    if final_path.exists():
        print("checkpoint: loading completed 81-phase rear audit", flush=True)
        return json.loads(final_path.read_text())

    partial_path = out / "rear_partial.json"
    rows = load_rows(partial_path)
    completed = {point_key(row) for row in rows}
    nominal = ParametricStudy(speed_kmh=70.0)
    fixed_gain = nominal.K.copy()
    all_points = points()
    start = time.time()

    for index, point in enumerate(all_points, 1):
        if point_key(point) in completed:
            continue
        study = study_at(
            point["speed_kmh"],
            point["mass_fraction"],
            point["payload_height"],
            fixed_gain,
            phases=9,
        )
        certificate = rear_pair_certificate(study, coefficients)
        baseline = certificate["baseline"]["grid_feasible_lower"]
        row = {
            **point,
            **certificate,
            "relative_margin": float(certificate["certified_margin"] / baseline),
            "relative_margin_percent": float(
                100.0 * certificate["certified_margin"] / baseline
            ),
        }
        rows.append(row)
        completed.add(point_key(point))
        if len(rows) % 10 == 0:
            base.save_json(partial_path, rows)
            print(f"rear checkpoint {len(rows)}/{len(all_points)}", flush=True)

    base.save_json(partial_path, rows)
    limiting = min(rows, key=lambda row: row["relative_margin"])
    record = {
        "phase_count": PHASES_REAR,
        "count": len(rows),
        "target_relative_margin": float(target),
        "minimum_absolute_margin": limiting["certified_margin"],
        "minimum_relative_margin": limiting["relative_margin"],
        "minimum_relative_margin_percent": limiting["relative_margin_percent"],
        "limiting_point": limiting,
        "all_non_degrading": all(row["certified_margin"] >= -2e-8 for row in rows),
        "target_achieved": all(
            row["relative_margin"] >= float(target) - 1e-10 for row in rows
        ),
        "rows": rows,
        "runtime_seconds_current_invocation": time.time() - start,
    }
    base.save_json(final_path, record)
    print("checkpoint: rear audit saved", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--guard", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 < args.target < 0.20:
        raise ValueError("target must be supplied as a fraction, e.g. 0.01")
    if args.guard <= 0.0:
        raise ValueError("guard must be positive")

    out = HERE / "outputs" / target_stem(args.target)
    out.mkdir(parents=True, exist_ok=True)
    design, coefficients = synthesize(args.target, args.guard, out)
    screening = broad_screen(coefficients, out)
    rear = fine_rear_audit(coefficients, args.target, out)

    status = "PASS" if (
        rear["target_achieved"]
        and rear["all_non_degrading"]
        and screening["max_constraint_utilization"] <= 1.0 + 2e-8
        and screening["max_spectral_radius"] < 1.0
    ) else "TARGET_MISSED"
    summary = {
        "status": status,
        "date": "2026-09-21",
        "target_relative_margin": args.target,
        "design_absolute_grid_guard": args.guard,
        "design_gamma": design["design_gamma"],
        "design_iterations": design["iterations"],
        "worst_front_LTR_21_phase": screening["worst_front_LTR"],
        "worst_LTR_21_phase": screening["worst_LTR"],
        "max_constraint_utilization_21_phase": screening["max_constraint_utilization"],
        "max_spectral_radius": screening["max_spectral_radius"],
        "minimum_absolute_rear_margin_81_phase": rear["minimum_absolute_margin"],
        "minimum_relative_rear_margin": rear["minimum_relative_margin"],
        "minimum_relative_rear_margin_percent": rear["minimum_relative_margin_percent"],
        "limiting_rear_point": {
            key: rear["limiting_point"][key] for key in
            ("set", "speed_kmh", "mass_fraction", "payload_height")
        },
        "target_achieved": rear["target_achieved"],
        "all_non_degrading": rear["all_non_degrading"],
    }
    base.save_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()