"""Fine-time verification of the stage-4 operating-envelope conclusions.

The broad screening uses 21 phases per 20 ms control interval.  This script
revisits all rear-axle comparisons with 81 phases and uses a one-sided proof:
the proposed continuous-time upper bound must not exceed the feasible sampled
lower bound for the LQR1 peak.  It also recomputes all outputs at the finite set
of parameter points that determine the reported worst cases.
"""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np

from study import ParametricStudy, base, compact_evaluate, study_at
from run_envelope import SPEED_NODES, scheduled_assets
from study import interpolate_schedule

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
PHASES = 81


def rear_pair_certificate(study, proposed):
    phase_times = np.linspace(0.0, study.h, PHASES)
    maps = study.maps(phase_times)
    zero = np.zeros(12)
    records = {}
    for name, coefficients in (("baseline", zero), ("proposed", proposed)):
        g = study.combine_maps(maps, coefficients)
        corrected, primal = [], []
        max_dual_correction = 0.0
        for phase in range(PHASES):
            upper, _, feasible_value, correction = study.support(
                g[phase, 1], return_witness=True
            )
            corrected.append(float(upper))
            primal.append(float(feasible_value))
            max_dual_correction = max(max_dual_correction, float(correction))
        spacing = study.h / (PHASES - 1)
        interpolation, _, _ = study._intersample_correction(coefficients, spacing)
        records[name] = {
            "grid_corrected_upper": max(corrected),
            "grid_feasible_lower": max(primal),
            "intersample_correction": float(interpolation[1]),
            "continuous_upper": max(corrected) + float(interpolation[1]),
            "max_dual_correction": max_dual_correction,
        }
    records["certified_margin"] = (
        records["baseline"]["grid_feasible_lower"]
        - records["proposed"]["continuous_upper"]
    )
    records["certified"] = records["certified_margin"] >= -2e-8
    return records


def point_key(row):
    return (
        round(float(row["speed_kmh"]), 12),
        round(float(row["mass_fraction"]), 12),
        round(float(row["payload_height"]), 12),
    )


def main():
    start = time.time()
    screening = json.loads((OUT / "envelope_results.json").read_text())
    nominal = ParametricStudy(speed_kmh=70.0)
    k70 = nominal.K.copy()
    fixed_balanced = np.loadtxt(
        ROOT / "certified_aarb_v3/outputs/bank_fixed_balanced_coefficients.csv",
        delimiter=",",
    ).reshape(-1)
    fixed_protected = np.loadtxt(
        OUT / "fixed_rear_protected_coefficients.csv", delimiter=",",
    ).reshape(-1)
    scheduled_gains, scheduled_coefficients = scheduled_assets()

    rear_rows = []
    for index, row in enumerate(screening["rows"], 1):
        study = study_at(
            row["speed_kmh"], row["mass_fraction"], row["payload_height"],
            k70, phases=9,
        )
        certificate = rear_pair_certificate(study, fixed_protected)
        rear_rows.append({
            "set": row["set"],
            "speed_kmh": row["speed_kmh"],
            "mass_fraction": row["mass_fraction"],
            "payload_height": row["payload_height"],
            **certificate,
        })
        if index % 10 == 0:
            print(f"rear verification {index}/{len(screening['rows'])}", flush=True)

    # Recompute all outputs only at the points that determine each headline
    # maximum, plus the least rear-axle margin from the fine verification.
    critical = {}
    for controller, info in screening["controllers"].items():
        for field in ("worst_LTR_point", "constraint_point", "spectral_radius_point"):
            point = info[field]
            critical[point_key(point)] = point
    baseline_point = screening["baseline_worst_point"]
    critical[point_key(baseline_point)] = baseline_point
    rear_limit = min(rear_rows, key=lambda r: r["certified_margin"])
    critical[point_key(rear_limit)] = rear_limit

    full_rows = []
    for key, point in critical.items():
        speed, fraction, height = key
        fixed_study = study_at(speed, fraction, height, k70, phases=9)
        scheduled_gain = interpolate_schedule(speed, SPEED_NODES, scheduled_gains)
        scheduled_f = interpolate_schedule(speed, SPEED_NODES, scheduled_coefficients)
        scheduled_study = study_at(
            speed, fraction, height, scheduled_gain, phases=9
        )
        row = {
            "speed_kmh": speed,
            "mass_fraction": fraction,
            "payload_height": height,
            "baseline": compact_evaluate(fixed_study, np.zeros(12), phases=PHASES),
            "fixed_balanced": compact_evaluate(
                fixed_study, fixed_balanced, phases=PHASES
            ),
            "fixed_rear_protected": compact_evaluate(
                fixed_study, fixed_protected, phases=PHASES
            ),
            "scheduled_balanced": compact_evaluate(
                scheduled_study, scheduled_f, phases=PHASES
            ),
        }
        full_rows.append(row)
        print("full verification", speed, fraction, height, flush=True)

    min_margin = min(r["certified_margin"] for r in rear_rows)
    result = {
        "status": "PASS" if min_margin >= -2e-8 else "REVIEW",
        "date": "2026-09-19",
        "phases": PHASES,
        "phase_spacing_ms": 1000.0 * 0.02 / (PHASES - 1),
        "rear_certificate_definition": (
            "proposed continuous-time upper <= baseline feasible sampled lower"
        ),
        "rear_point_count": len(rear_rows),
        "minimum_certified_margin": min_margin,
        "limiting_rear_point": min(rear_rows, key=lambda r: r["certified_margin"]),
        "rear_all_certified": all(r["certified"] for r in rear_rows),
        "rear_rows": rear_rows,
        "critical_full_output_rows": full_rows,
        "runtime_seconds": time.time() - start,
    }
    base.save_json(OUT / "envelope_verification.json", result)
    print(json.dumps({k: result[k] for k in (
        "status", "phases", "rear_point_count", "minimum_certified_margin",
        "limiting_rear_point", "rear_all_certified", "runtime_seconds",
    )}, indent=2))


if __name__ == "__main__":
    main()