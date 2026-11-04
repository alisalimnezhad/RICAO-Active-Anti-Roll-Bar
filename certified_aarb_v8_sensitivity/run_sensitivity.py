"""Checkpointed fixed-controller FIR, horizon, steering and hydraulic audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from study import (
    ConfigurableStudy, all_operating_points, base, build_study,
    evaluate_point, p, skeleton_points, structured_points,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
PARTIAL = OUT / "partial.json"
RESULTS = OUT / "results.json"
COEFFICIENTS = (
    ROOT / "certified_aarb_v5_rear_audit/outputs/comparison/"
    "selected_reserve_5pct_coefficients.csv"
)
BROAD_PHASES = 21
REAR_PHASES = 41


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fir_coefficients(full, taps):
    matrix = np.asarray(full).reshape(2, 6).copy()
    keep = {1: 1, 2: 2, 3: 3, 4: 4, 6: 6}[int(taps)]
    matrix[:, keep:] = 0.0
    return matrix.reshape(-1)


def case_definitions(full):
    cases = []
    for taps in (1, 2, 3, 4, 6):
        cases.append({
            "key": f"fir_{taps}", "group": "fir", "bank": "structured42",
            "label": f"FIR-{taps}", "taps": taps,
            "coefficients": fir_coefficients(full, taps).tolist(),
            "n": 200, "amp_deg": 2.0, "rate_deg_s": 5.0,
            "extra_overrides": {},
        })
    for horizon in (2.0, 4.0, 6.0):
        cases.append({
            "key": f"horizon_{horizon:g}s", "group": "horizon",
            "bank": "structured42", "label": f"T={horizon:g} s",
            "horizon_s": horizon, "coefficients": full.tolist(),
            "n": int(round(horizon / base.H)), "amp_deg": 2.0,
            "rate_deg_s": 5.0, "extra_overrides": {},
        })
    for amp in (1.0, 2.0, 3.0):
        for rate in (2.5, 5.0, 7.5):
            cases.append({
                "key": f"steering_{amp:g}deg_{rate:g}dps",
                "group": "steering", "bank": "structured42",
                "label": f"{amp:g} deg; {rate:g} deg/s",
                "coefficients": full.tolist(), "n": 200,
                "amp_deg": amp, "rate_deg_s": rate,
                "extra_overrides": {},
            })

    hydraulic = (
        ("Kv", p.Kv), ("tau", p.tau), ("Kx", p.Kx), ("KP", p.KP),
        ("bulk", p.bulk), ("Vt", p.Vt), ("area", p.area),
    )
    for name, nominal in hydraulic:
        for factor in (0.8, 0.9, 1.1, 1.2):
            cases.append({
                "key": f"hyd_{name}_{factor:.1f}", "group": "hydraulic",
                "bank": "skeleton9", "label": f"{name} {factor:.1f}x",
                "parameter": name, "factor": factor,
                "coefficients": full.tolist(), "n": 200,
                "amp_deg": 2.0, "rate_deg_s": 5.0,
                "extra_overrides": {name: nominal * factor},
            })
    for leakage_ratio in (0.1, 0.2):
        cases.append({
            "key": f"hyd_Ctp_{leakage_ratio:.1f}KP", "group": "hydraulic",
            "bank": "skeleton9", "label": f"Ctp={leakage_ratio:.1f} KP",
            "parameter": "Ctp", "factor": leakage_ratio,
            "coefficients": full.tolist(), "n": 200,
            "amp_deg": 2.0, "rate_deg_s": 5.0,
            "extra_overrides": {"Ctp": p.KP * leakage_ratio},
        })
    cases.append({
        "key": "hyd_nominal", "group": "hydraulic", "bank": "skeleton9",
        "label": "hydraulic nominal", "parameter": "nominal", "factor": 1.0,
        "coefficients": full.tolist(), "n": 200, "amp_deg": 2.0,
        "rate_deg_s": 5.0, "extra_overrides": {},
    })
    return cases


def summarize_case(case, point_rows):
    stable = [row for row in point_rows if row["evaluation"]["status"] == "PASS"]
    status = "PASS" if len(stable) == len(point_rows) else "FAIL_UNSTABLE"
    summary = {
        "key": case["key"], "group": case["group"], "bank": case["bank"],
        "label": case["label"], "status": status,
        "point_count": len(point_rows), "stable_count": len(stable),
        "online_fir_coefficients": 2 * int(case.get("taps", 6)),
        "redesign_performed": False, "points": point_rows,
    }
    for key in ("taps", "horizon_s", "amp_deg", "rate_deg_s", "parameter", "factor"):
        if key in case:
            summary[key] = case[key]
    rho_row = max(point_rows, key=lambda row: row["evaluation"]["spectral_radius"])
    summary["max_spectral_radius"] = rho_row["evaluation"]["spectral_radius"]
    summary["spectral_radius_point"] = rho_row["point"]
    if stable:
        front = max(stable, key=lambda row: row["evaluation"]["front_LTR"])
        rear = max(stable, key=lambda row: row["evaluation"]["rear_LTR"])
        worst = max(stable, key=lambda row: row["evaluation"]["worst_LTR"])
        util = max(stable, key=lambda row: row["evaluation"]["max_constraint_utilization"])
        margin = min(stable, key=lambda row: row["evaluation"]["rear_margin"]["relative_margin"])
        summary.update({
            "worst_front_LTR": front["evaluation"]["front_LTR"],
            "worst_front_point": front["point"],
            "worst_rear_LTR": rear["evaluation"]["rear_LTR"],
            "worst_rear_point": rear["point"],
            "worst_LTR": worst["evaluation"]["worst_LTR"],
            "worst_LTR_point": worst["point"],
            "max_constraint_utilization": util["evaluation"]["max_constraint_utilization"],
            "limiting_constraint": util["evaluation"]["limiting_constraint"],
            "constraint_point": util["point"],
            "minimum_rear_margin_percent": margin["evaluation"]["rear_margin"]["relative_margin_percent"],
            "minimum_rear_margin_point": margin["point"],
            "constraint_feasible": all(
                row["evaluation"]["max_constraint_utilization"] <= 1.0 + 2e-8
                for row in stable
            ),
            "rear_non_degrading": all(
                row["evaluation"]["rear_margin"]["certified_margin"] >= -2e-8
                for row in stable
            ),
        })
        if status == "PASS" and not summary["constraint_feasible"]:
            status = "FAIL_CONSTRAINT"
        elif status == "PASS" and not summary["rear_non_degrading"]:
            status = "FAIL_REAR_DEGRADATION"
        summary["status"] = status
    return summary


def reclassify(record):
    """Apply the final status convention to checkpointed case summaries."""
    if record["stable_count"] != record["point_count"]:
        record["status"] = "FAIL_UNSTABLE"
    elif not record.get("constraint_feasible", False):
        record["status"] = "FAIL_CONSTRAINT"
    elif not record.get("rear_non_degrading", False):
        record["status"] = "FAIL_REAR_DEGRADATION"
    else:
        record["status"] = "PASS"
    return record


def delay_summary(feedback_gain):
    points = all_operating_points()
    rows = []
    for delay_ms in (0.0, 5.0, 10.0, 20.0):
        radii = []
        for point in points:
            study = build_study(point, feedback_gain)
            delay_s = delay_ms / 1000.0
            elapsed = study.h - delay_s
            Aa, Ba, _, _ = base.foh_zoh_lift(
                study.A, study.B, study.D, elapsed, study.h
            )
            augmented = np.block([
                [study.ad, -study.bd_phys @ study.K],
                [Aa, -Ba @ study.K],
            ])
            radii.append(float(np.max(np.abs(np.linalg.eigvals(augmented)))))
        rows.append({
            "delay_ms": delay_ms, "plant_count": len(radii),
            "minimum_spectral_radius": min(radii),
            "maximum_spectral_radius": max(radii),
            "stable_count": sum(value < 1.0 for value in radii),
            "status": "PASS" if all(value < 1.0 for value in radii) else "FAIL_UNSTABLE",
            "redesign_performed": False,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-cases", type=int, default=None)
    args = parser.parse_args()
    start = time.time()
    full = np.loadtxt(COEFFICIENTS, delimiter=",").reshape(-1)
    nominal = ConfigurableStudy(speed_kmh=70.0, phases=9)
    feedback_gain = nominal.K.copy()
    cases = case_definitions(full)
    records = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else []
    completed = {row["key"] for row in records}
    new_count = 0
    banks = {"structured42": structured_points(), "skeleton9": skeleton_points()}

    for case in cases:
        if case["key"] in completed:
            continue
        point_rows = []
        for index, point in enumerate(banks[case["bank"]], 1):
            study = build_study(
                point, feedback_gain, n=case["n"],
                amp=np.deg2rad(case["amp_deg"]),
                rate=np.deg2rad(case["rate_deg_s"]),
                extra_overrides=case["extra_overrides"],
            )
            point_rows.append({
                "point": dict(point),
                "evaluation": evaluate_point(
                    study, np.asarray(case["coefficients"]),
                    broad_phases=BROAD_PHASES, rear_phases=REAR_PHASES,
                ),
            })
        record = summarize_case(case, point_rows)
        records.append(record)
        completed.add(case["key"])
        save(PARTIAL, records)
        new_count += 1
        print(
            f"sensitivity case {len(records)}/{len(cases)}: {case['key']} "
            f"status={record['status']}", flush=True
        )
        if args.max_new_cases is not None and new_count >= args.max_new_cases:
            break

    if len(records) < len(cases):
        print(json.dumps({"status": "PARTIAL", "completed": len(records),
                          "total": len(cases)}, indent=2))
        return

    records = [reclassify(record) for record in records]
    save(PARTIAL, records)
    delays = delay_summary(feedback_gain)
    result = {
        "status": "COMPLETE",
        "date": "2026-09-23",
        "protocol": {
            "controller": "selected fixed 5% rear-reserve controller",
            "controller_redesigned_in_sensitivity": False,
            "broad_phases": BROAD_PHASES,
            "rear_screen_phases": REAR_PHASES,
            "fir_horizon_steering_bank": "42 structured operating points",
            "hydraulic_bank": "9 design-skeleton operating points",
            "delay_bank": "42 structured plus 128 Sobol points",
            "fixed_physical_steering_gain": True,
        },
        "cases": records,
        "delay": delays,
        "source_hashes": {
            "run_sensitivity.py": sha256(Path(__file__)),
            "study.py": sha256(HERE / "study.py"),
            "selected_coefficients": sha256(COEFFICIENTS),
            "v4_study.py": sha256(ROOT / "certified_aarb_v4/study.py"),
            "vehicle_model.py": sha256(ROOT / "validation/vehicle_model.py"),
            "parameters.py": sha256(ROOT / "validation/parameters.py"),
        },
        "runtime_seconds": time.time() - start,
        "limitations": [
            "This is fixed-controller sensitivity; no controller is redesigned.",
            "FIR, horizon and steering use the 42-point structured bank.",
            "Hydraulic perturbations use the nine-point design skeleton.",
            "The calculations are pointwise in vehicle and hydraulic parameters.",
        ],
    }
    save(RESULTS, result)
    print(json.dumps({"status": result["status"], "cases": len(records),
                      "delay_cases": len(delays),
                      "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()