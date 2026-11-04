"""Checkpointed 5 ms sensor/computation-delay audit over 170 plants."""
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
V4 = ROOT / "certified_aarb_v4"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))

from delay_study import DelayedParametricStudy  # noqa: E402
from study import base, compact_evaluate, distributed_payload_overrides  # noqa: E402
from run_envelope import controller_summary, points  # noqa: E402


DELAY_S = 0.005
PHASES_SCREEN = 21
PHASES_REAR = 81
OUT = HERE / "outputs" / "delay_sensitivity" / "delay_5ms"
OUT.mkdir(parents=True, exist_ok=True)
PARTIAL = OUT / "partial.json"
FINAL = OUT / "results.json"
CURRENT_COEFFICIENTS = V4 / "outputs" / "fixed_rear_protected_coefficients.csv"
RESERVE_COEFFICIENTS = HERE / "outputs" / "reserve_5pct" / "coefficients.csv"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return (
        round(float(row["speed_kmh"]), 12),
        round(float(row["mass_fraction"]), 12),
        round(float(row["payload_height"]), 12),
        row["set"],
    )


def delayed_study(point, gain):
    return DelayedParametricStudy(
        speed_kmh=point["speed_kmh"],
        overrides=distributed_payload_overrides(
            point["mass_fraction"], point["payload_height"]
        ),
        feedback_gain=gain,
        delay_s=DELAY_S,
        phases=9,
        label=(
            f"5 ms; v={point['speed_kmh']:.6g}; "
            f"mu={point['mass_fraction']:.6g}; hp={point['payload_height']:.6g}"
        ),
    )


def rear_records(study, coefficients):
    phase_times = np.linspace(0.0, study.h, PHASES_REAR)
    maps = study.maps(phase_times)
    records = {}
    for name, f in coefficients.items():
        f = np.asarray(f, dtype=float)
        g = study.combine_maps(maps, f)
        corrected, feasible = [], []
        max_correction = 0.0
        for phase in range(PHASES_REAR):
            upper, _, primal, correction = study.support(
                g[phase, 1], return_witness=True
            )
            corrected.append(float(upper))
            feasible.append(float(primal))
            max_correction = max(max_correction, float(correction))
        spacing = study.h / (PHASES_REAR - 1)
        interpolation, _, _ = study._intersample_correction(f, spacing)
        records[name] = {
            "grid_corrected_upper": max(corrected),
            "grid_feasible_lower": max(feasible),
            "intersample_correction": float(interpolation[1]),
            "continuous_upper": max(corrected) + float(interpolation[1]),
            "max_dual_correction": max_correction,
        }
    baseline = records["baseline"]["grid_feasible_lower"]
    for name in ("current", "reserve_5pct"):
        margin = baseline - records[name]["continuous_upper"]
        records[name]["certified_margin"] = margin
        records[name]["relative_margin"] = margin / baseline
        records[name]["relative_margin_percent"] = 100.0 * margin / baseline
        records[name]["non_degrading"] = margin >= -2e-8
    return records


def point_dict(row):
    return {name: row[name] for name in (
        "set", "speed_kmh", "mass_fraction", "payload_height"
    )}


def summarize(rows):
    current_front = max(rows, key=lambda row: row["current"]["front_LTR"])
    reserve_front = max(rows, key=lambda row: row["reserve_5pct"]["front_LTR"])
    current_util = max(
        rows, key=lambda row: row["current"]["constraint_utilization"]
    )
    reserve_util = max(
        rows, key=lambda row: row["reserve_5pct"]["constraint_utilization"]
    )
    current_margin = min(
        rows, key=lambda row: row["rear_certificate"]["current"]["relative_margin"]
    )
    reserve_margin = min(
        rows, key=lambda row: row["rear_certificate"]["reserve_5pct"]["relative_margin"]
    )
    rho_row = max(rows, key=lambda row: row["spectral_radius"])
    min_rho_row = min(rows, key=lambda row: row["spectral_radius"])
    return {
        "status": "FAIL_UNSTABLE" if any(
            row["spectral_radius"] >= 1.0 for row in rows
        ) else "PASS",
        "date": "2026-09-22",
        "delay_ms": 5.0,
        "plant_count": len(rows),
        "counts_by_set": {
            name: sum(row["set"] == name for row in rows)
            for name in ("structured", "sobol_holdout")
        },
        "protocol": {
            "controller_period_ms": 20.0,
            "common_state_and_steering_delay_ms": 5.0,
            "current_hold": "zero order",
            "physical_steering_hold": "first order",
            "broad_certificate_phases": PHASES_SCREEN,
            "rear_certificate_phases": PHASES_REAR,
            "rear_comparator": "LQR1 with the same 5 ms measurement delay",
        },
        "stability": {
            "minimum_spectral_radius": min_rho_row["spectral_radius"],
            "minimum_point": point_dict(min_rho_row),
            "maximum_spectral_radius": rho_row["spectral_radius"],
            "maximum_point": point_dict(rho_row),
            "stable_count": sum(row["spectral_radius"] < 1.0 for row in rows),
            "unstable_count": sum(row["spectral_radius"] >= 1.0 for row in rows),
        },
        "current": {
            "worst_front_LTR": current_front["current"]["front_LTR"],
            "worst_front_point": point_dict(current_front),
            "max_constraint_utilization": current_util["current"]["constraint_utilization"],
            "limiting_constraint": current_util["current"]["limiting_constraint"],
            "constraint_point": point_dict(current_util),
            "minimum_rear_margin_percent": current_margin["rear_certificate"]["current"]["relative_margin_percent"],
            "limiting_rear_point": point_dict(current_margin),
            "non_degrading_count": sum(
                row["rear_certificate"]["current"]["non_degrading"] for row in rows
            ),
            "constraint_feasible_count": sum(
                row["current"]["constraint_utilization"] <= 1.0 + 2e-8
                for row in rows
            ),
        },
        "reserve_5pct": {
            "worst_front_LTR": reserve_front["reserve_5pct"]["front_LTR"],
            "worst_front_point": point_dict(reserve_front),
            "max_constraint_utilization": reserve_util["reserve_5pct"]["constraint_utilization"],
            "limiting_constraint": reserve_util["reserve_5pct"]["limiting_constraint"],
            "constraint_point": point_dict(reserve_util),
            "minimum_rear_margin_percent": reserve_margin["rear_certificate"]["reserve_5pct"]["relative_margin_percent"],
            "limiting_rear_point": point_dict(reserve_margin),
            "non_degrading_count": sum(
                row["rear_certificate"]["reserve_5pct"]["non_degrading"] for row in rows
            ),
            "five_percent_count": sum(
                row["rear_certificate"]["reserve_5pct"]["relative_margin"] >= 0.05
                for row in rows
            ),
            "constraint_feasible_count": sum(
                row["reserve_5pct"]["constraint_utilization"] <= 1.0 + 2e-8
                for row in rows
            ),
        },
        "scope": (
            "Exact finite-horizon steering/time calculations at the evaluated "
            "linear plants under the declared uniform 5 ms measurement delay; "
            "not an interval proof between parameter points."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-points", type=int, default=None)
    args = parser.parse_args()
    start = time.time()
    current_f = np.loadtxt(CURRENT_COEFFICIENTS, delimiter=",").reshape(-1)
    reserve_f = np.loadtxt(RESERVE_COEFFICIENTS, delimiter=",").reshape(-1)
    coefficients = {
        "baseline": np.zeros(12),
        "current": current_f,
        "reserve_5pct": reserve_f,
    }
    nominal = DelayedParametricStudy(speed_kmh=70.0, delay_s=0.0, phases=9)
    fixed_gain = nominal.K.copy()
    rows = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else []
    completed = {key(row) for row in rows}
    new_count = 0

    for point in points():
        if key(point) in completed:
            continue
        study = delayed_study(point, fixed_gain)
        current_eval = compact_evaluate(study, current_f, phases=PHASES_SCREEN)
        reserve_eval = compact_evaluate(study, reserve_f, phases=PHASES_SCREEN)
        row = {
            **point,
            "delay_ms": 5.0,
            "spectral_radius": float(study.rho_closed),
            "current": controller_summary(study, current_eval),
            "reserve_5pct": controller_summary(study, reserve_eval),
            "rear_certificate": rear_records(study, coefficients),
        }
        rows.append(row)
        completed.add(key(point))
        new_count += 1
        base.save_json(PARTIAL, rows)
        print(
            f"delay 5 ms checkpoint {len(rows)}/170; "
            f"rho={study.rho_closed:.6f}", flush=True
        )
        if args.max_new_points is not None and new_count >= args.max_new_points:
            break

    if len(rows) < 170:
        print(json.dumps({
            "status": "PARTIAL", "completed": len(rows),
            "runtime_seconds_current_invocation": time.time() - start,
        }, indent=2))
        return

    summary = summarize(rows)
    summary["runtime_seconds_current_invocation"] = time.time() - start
    summary["source_hashes"] = {
        "delay_study.py": sha256(HERE / "delay_study.py"),
        "run_delay_5ms.py": sha256(Path(__file__)),
        "v4_study.py": sha256(V4 / "study.py"),
        "v3_study.py": sha256(ROOT / "certified_aarb_v3" / "study.py"),
        "v2_study.py": sha256(ROOT / "certified_aarb_v2" / "study.py"),
        "current_coefficients.csv": sha256(CURRENT_COEFFICIENTS),
        "reserve_coefficients.csv": sha256(RESERVE_COEFFICIENTS),
    }
    base.save_json(FINAL, {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()