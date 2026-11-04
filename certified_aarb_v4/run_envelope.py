"""Evaluate the stage-4 controllers on structured and unseen parameters."""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
from scipy.stats import qmc

from study import (
    ParametricStudy, base, compact_evaluate, distributed_payload_overrides,
    interpolate_schedule, selected_evaluate, study_at, p,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

PHASES = 21
SPEED_NODES = np.array([50.0, 70.0, 75.0])


def scheduled_assets():
    gains = [ParametricStudy(speed_kmh=speed).K.copy() for speed in SPEED_NODES]
    coefficients = [
        np.asarray(json.loads((
            ROOT / "certified_aarb_v3/outputs" /
            f"bank_scheduled_{str(speed).replace('.', 'p')}.json"
        ).read_text())["coefficients"], dtype=float)
        for speed in SPEED_NODES
    ]
    return gains, coefficients


def points():
    structured = []
    for speed in np.linspace(50.0, 75.0, 6):
        structured.append({"set": "structured", "speed_kmh": speed,
                           "mass_fraction": 0.0, "payload_height": p.h})
        for fraction in (0.05, 0.10):
            for height in (p.h, 1.475, 1.80):
                structured.append({"set": "structured", "speed_kmh": speed,
                                   "mass_fraction": fraction,
                                   "payload_height": height})
    sampler = qmc.Sobol(d=3, scramble=True, seed=20260919)
    unit = sampler.random_base2(m=7)
    holdout = []
    for row in unit:
        holdout.append({
            "set": "sobol_holdout",
            "speed_kmh": 50.0 + 25.0 * row[0],
            "mass_fraction": 0.10 * row[1],
            "payload_height": p.h + (1.80 - p.h) * row[2],
        })
    return structured + holdout


def controller_summary(study, result):
    values = result["continuous"]
    front = float(values["front_LTR"])
    rear = float(values["rear_LTR"])
    non_ltr = {k: float(v) for k, v in values.items()
               if k not in ("front_LTR", "rear_LTR")}
    if non_ltr:
        limiting = max(non_ltr, key=non_ltr.get)
        utilization = non_ltr[limiting]
    else:
        limiting, utilization = None, None
    return {
        "front_LTR": front,
        "rear_LTR": rear,
        "worst_LTR": max(front, rear),
        "constraint_utilization": utilization,
        "limiting_constraint": limiting,
        "spectral_radius": float(study.rho_closed),
        "max_support_correction": float(result["max_support_correction"]),
    }


def main():
    start = time.time()
    nominal = ParametricStudy(speed_kmh=70.0)
    k70 = nominal.K.copy()
    zero = np.zeros(12)
    fixed_balanced = np.loadtxt(
        ROOT / "certified_aarb_v3/outputs/bank_fixed_balanced_coefficients.csv",
        delimiter=",",
    ).reshape(-1)
    fixed_protected = np.loadtxt(
        OUT / "fixed_rear_protected_coefficients.csv", delimiter=",",
    ).reshape(-1)
    scheduled_gains, scheduled_coefficients = scheduled_assets()

    protocol = {
        "date": "2026-09-19",
        "phases": PHASES,
        "phase_spacing_ms": 1000.0 * 0.02 / (PHASES - 1),
        "speed_interval_kmh": [50.0, 75.0],
        "added_mass_fraction_interval": [0.0, 0.10],
        "payload_height_interval_m": [p.h, 1.80],
        "structured_grid": {
            "speeds_kmh": np.linspace(50.0, 75.0, 6),
            "mass_fractions": [0.0, 0.05, 0.10],
            "payload_heights_m_for_loaded_cases": [p.h, 1.475, 1.80],
            "count": 42,
        },
        "holdout": {"method": "scrambled Sobol", "seed": 20260919,
                    "count": 128},
        "schedule_nodes_kmh": SPEED_NODES,
        "claim": (
            "Exact steering/time certificate at every evaluated parameter point; "
            "structured plus unseen-parameter verification, not an interval proof "
            "between parameter points."
        ),
    }
    base.save_json(OUT / "envelope_protocol.json", protocol)

    all_points = points()
    partial_path = OUT / "envelope_partial.json"
    rows = json.loads(partial_path.read_text()) if partial_path.exists() else []
    completed = {(round(r["speed_kmh"], 12), round(r["mass_fraction"], 12),
                  round(r["payload_height"], 12), r["set"]) for r in rows}

    for index, point in enumerate(all_points, 1):
        key = (round(point["speed_kmh"], 12), round(point["mass_fraction"], 12),
               round(point["payload_height"], 12), point["set"])
        if key in completed:
            continue
        overrides = distributed_payload_overrides(
            point["mass_fraction"], point["payload_height"]
        )
        fixed_study = study_at(
            point["speed_kmh"], point["mass_fraction"],
            point["payload_height"], k70, phases=9,
        )
        baseline_result = selected_evaluate(
            fixed_study, zero,
            outputs=(0, 1, 2, 3, 12, 13, 14, 15), phases=PHASES,
        )
        balanced_result = compact_evaluate(
            fixed_study, fixed_balanced, phases=PHASES
        )
        protected_result = compact_evaluate(
            fixed_study, fixed_protected, phases=PHASES
        )

        scheduled_gain = interpolate_schedule(
            point["speed_kmh"], SPEED_NODES, scheduled_gains
        )
        scheduled_f = interpolate_schedule(
            point["speed_kmh"], SPEED_NODES, scheduled_coefficients
        )
        scheduled_study = study_at(
            point["speed_kmh"], point["mass_fraction"],
            point["payload_height"], scheduled_gain, phases=9,
        )
        scheduled_result = compact_evaluate(
            scheduled_study, scheduled_f, phases=PHASES
        )
        physical = distributed_payload_overrides(
            point["mass_fraction"], point["payload_height"]
        )
        row = dict(point)
        row["derived"] = physical
        row["baseline"] = controller_summary(fixed_study, baseline_result)
        row["fixed_balanced"] = controller_summary(fixed_study, balanced_result)
        row["fixed_rear_protected"] = controller_summary(fixed_study, protected_result)
        row["scheduled_balanced"] = controller_summary(scheduled_study, scheduled_result)
        row["rear_ratio_protected_to_baseline"] = (
            row["fixed_rear_protected"]["rear_LTR"] /
            row["baseline"]["rear_LTR"]
        )
        row["schedule_improvement_percent"] = 100.0 * (
            1.0 - row["scheduled_balanced"]["worst_LTR"] /
            row["fixed_balanced"]["worst_LTR"]
        )
        rows.append(row)
        completed.add(key)
        if len(rows) % 5 == 0:
            base.save_json(partial_path, rows)
            print(f"completed {len(rows)}/{len(all_points)}", flush=True)

    base.save_json(partial_path, rows)
    controllers = ("fixed_balanced", "fixed_rear_protected", "scheduled_balanced")
    worst = {}
    for controller in controllers:
        worst_row = max(rows, key=lambda r: r[controller]["worst_LTR"])
        util_row = max(rows, key=lambda r: r[controller]["constraint_utilization"])
        rho_row = max(rows, key=lambda r: r[controller]["spectral_radius"])
        worst[controller] = {
            "worst_LTR": worst_row[controller]["worst_LTR"],
            "worst_LTR_point": {k: worst_row[k] for k in
                                 ("set", "speed_kmh", "mass_fraction", "payload_height")},
            "max_constraint_utilization": util_row[controller]["constraint_utilization"],
            "limiting_constraint": util_row[controller]["limiting_constraint"],
            "constraint_point": {k: util_row[k] for k in
                                  ("set", "speed_kmh", "mass_fraction", "payload_height")},
            "max_spectral_radius": rho_row[controller]["spectral_radius"],
            "spectral_radius_point": {k: rho_row[k] for k in
                                      ("set", "speed_kmh", "mass_fraction", "payload_height")},
        }
    baseline_worst = max(rows, key=lambda r: r["baseline"]["worst_LTR"])
    rear_ratio_row = max(rows, key=lambda r: r["rear_ratio_protected_to_baseline"])
    summary = {
        "status": "PASS",
        "protocol": protocol,
        "count": len(rows),
        "counts_by_set": {
            name: sum(r["set"] == name for r in rows)
            for name in ("structured", "sobol_holdout")
        },
        "baseline_worst_LTR": baseline_worst["baseline"]["worst_LTR"],
        "baseline_worst_point": {k: baseline_worst[k] for k in
                                 ("set", "speed_kmh", "mass_fraction", "payload_height")},
        "controllers": worst,
        "max_rear_ratio_protected_to_baseline": rear_ratio_row["rear_ratio_protected_to_baseline"],
        "max_rear_ratio_point": {k: rear_ratio_row[k] for k in
                                 ("set", "speed_kmh", "mass_fraction", "payload_height")},
        "checks": {
            "fixed_balanced_constraints": all(r["fixed_balanced"]["constraint_utilization"] <= 1.0 + 2e-8 for r in rows),
            "fixed_protected_constraints": all(r["fixed_rear_protected"]["constraint_utilization"] <= 1.0 + 2e-8 for r in rows),
            "scheduled_constraints": all(r["scheduled_balanced"]["constraint_utilization"] <= 1.0 + 2e-8 for r in rows),
            "fixed_protected_rear_non_degradation": all(r["rear_ratio_protected_to_baseline"] <= 1.0 + 2e-8 for r in rows),
            "fixed_stability": all(r["fixed_rear_protected"]["spectral_radius"] < 1.0 for r in rows),
            "scheduled_stability": all(r["scheduled_balanced"]["spectral_radius"] < 1.0 for r in rows),
        },
        "rows": rows,
        "runtime_seconds": time.time() - start,
    }
    if not all(summary["checks"].values()):
        summary["status"] = "REVIEW"
    base.save_json(OUT / "envelope_results.json", summary)
    print(json.dumps({k: summary[k] for k in (
        "status", "count", "counts_by_set", "baseline_worst_LTR",
        "controllers", "max_rear_ratio_protected_to_baseline", "checks",
        "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()