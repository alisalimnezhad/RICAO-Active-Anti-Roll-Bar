"""Run Jacobian-gated nonlinear screening on critical 5 ms cases."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from delay_design import base, delayed_study, points
from nonlinear_screen import NonlinearPlant, simulate


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "nonlinear"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = HERE / "outputs" / "results.json"
GAIN_SCAN = HERE / "outputs" / "gain_scan.json"
COEFFICIENTS = HERE / "outputs" / "coefficients.csv"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point_key(point):
    return (
        round(float(point["speed_kmh"]), 12),
        round(float(point["mass_fraction"]), 12),
        round(float(point["payload_height"]), 12),
    )


def point_by_values(speed, fraction, height):
    for point in points():
        if point_key(point) == point_key({
            "speed_kmh": speed, "mass_fraction": fraction,
            "payload_height": height,
        }):
            return point
    return {
        "set": "critical",
        "speed_kmh": speed,
        "mass_fraction": fraction,
        "payload_height": height,
    }


def critical_witness(study, coefficients, output, phases=41):
    times = np.linspace(0.0, study.h, int(phases))
    maps = study.maps(times)
    g = study.combine_maps(maps, coefficients)
    best = None
    for phase in range(int(phases)):
        upper, witness, primal, correction = study.support(
            g[phase, int(output)], return_witness=True
        )
        if best is None or upper > best["corrected_upper"]:
            best = {
                "phase": phase,
                "phase_time_s": float(times[phase]),
                "corrected_upper": float(upper),
                "primal": float(primal),
                "dual_correction": float(correction),
                "witness": np.asarray(witness),
            }
    return best


def fishhook(n, rho):
    targets = [(0, 0.0), (40, 1.0), (90, -1.0), (140, 1.0), (199, 0.0)]
    w = np.zeros(n)
    target_index = 1
    value = 0.0
    for k in range(n):
        while target_index < len(targets) - 1 and k > targets[target_index][0]:
            target_index += 1
        target = targets[target_index][1]
        value += np.clip(target - value, -rho, rho)
        w[k] = value
    return w


def summarize_trajectory(trajectory):
    y = trajectory["outputs"]
    x = trajectory["state"]
    raw = trajectory["raw_current"]
    current = trajectory["current"]
    plant = trajectory["plant"]
    front, rear = [], []
    tyre_ratios = []
    if plant is not None:
        for state, delta in zip(x, trajectory["steering"]):
            rf, rr = plant.load_transfer(state)
            front.append(rf); rear.append(rr)
            fyf, fyr, capf, capr = plant.tyre_forces(state, delta)
            tyre_ratios.append(max(abs(fyf)/capf, abs(fyr)/capr))
    else:
        front = y[:, 0]; rear = y[:, 1]
    return {
        "peak_front_LTR": float(np.max(np.abs(front))),
        "peak_rear_LTR": float(np.max(np.abs(rear))),
        "peak_current_A": float(np.max(np.abs(current))),
        "peak_raw_current_A": float(np.max(np.abs(raw))),
        "current_saturation_fraction": float(np.mean(np.abs(raw) > base.I0 + 1e-12)),
        "peak_pressure_Pa": float(np.max(np.abs(x[:, [6, 8]]))),
        "peak_spool_m": float(np.max(np.abs(x[:, [7, 9]]))),
        "peak_tyre_capacity_fraction": (
            float(np.max(tyre_ratios)) if tyre_ratios else None
        ),
        "wheel_lift_proxy": bool(max(np.max(np.abs(front)), np.max(np.abs(rear))) >= 1.0),
        "finite": bool(np.isfinite(x).all() and np.isfinite(y).all()),
    }


def main():
    linear_results = json.loads(RESULTS.read_text())
    gain_record = json.loads(GAIN_SCAN.read_text())
    nominal_gain = np.asarray(gain_record["nominal_gain"])
    gain = float(gain_record["selected_scale"]) * nominal_gain
    coefficients = np.loadtxt(COEFFICIENTS, delimiter=",").reshape(-1)
    rows = linear_results["rows"]
    worst = max(rows, key=lambda row: row["delay_aware"]["worst_LTR"])
    constraint = max(
        rows, key=lambda row: row["delay_aware"]["constraint_utilization"]
    )
    rear_margin = min(
        rows,
        key=lambda row: row["rear_certificate_81_phase"]["relative_margin_percent"],
    )
    cases = [
        ("worst_front_witness", worst, 0),
        ("worst_rear_witness", worst, 1),
        ("limiting_front_tyre_slip_witness", constraint, 14),
        ("limiting_payload_rear_witness", rear_margin, 1),
    ]
    records = []
    jacobians = []
    for name, point, output in cases:
        study = delayed_study(point, gain, phases=5)
        audit = NonlinearPlant(study).jacobian_audit()
        jacobians.append({"case": name, **audit})
        if max(audit["relative_A"], audit["relative_B"], audit["relative_D"]) > 2e-5:
            raise RuntimeError(f"Jacobian audit failed for {name}: {audit}")
        witness = critical_witness(study, coefficients, output)
        linear = simulate(study, coefficients, witness["witness"], nonlinear=False)
        nonlinear = simulate(study, coefficients, witness["witness"], nonlinear=True)
        linear_summary = summarize_trajectory(linear)
        nonlinear_summary = summarize_trajectory(nonlinear)
        record = {
            "case": name,
            "source_output": study.names[output],
            "point": {k: point[k] for k in (
                "set", "speed_kmh", "mass_fraction", "payload_height"
            )},
            "witness_support": {k: v for k, v in witness.items() if k != "witness"},
            "linear": linear_summary,
            "nonlinear": nonlinear_summary,
            "front_peak_change_percent": 100.0 * (
                nonlinear_summary["peak_front_LTR"] /
                max(linear_summary["peak_front_LTR"], 1e-30) - 1.0
            ),
            "rear_peak_change_percent": 100.0 * (
                nonlinear_summary["peak_rear_LTR"] /
                max(linear_summary["peak_rear_LTR"], 1e-30) - 1.0
            ),
        }
        np.savez_compressed(
            OUT / f"{name}.npz",
            steering_witness=witness["witness"],
            time=nonlinear["time"],
            steering=nonlinear["steering"],
            linear_state=linear["state"],
            nonlinear_state=nonlinear["state"],
            linear_current=linear["current"],
            nonlinear_current=nonlinear["current"],
            linear_outputs=linear["outputs"],
            nonlinear_outputs=nonlinear["outputs"],
        )
        records.append(record)
        print(name, json.dumps(record["nonlinear"]), flush=True)

    fish_point = point_by_values(75.0, 0.10, 1.80)
    fish_study = delayed_study(fish_point, gain, phases=5)
    fish_w = fishhook(fish_study.n, base.RHO)
    fish_linear = simulate(fish_study, coefficients, fish_w, nonlinear=False)
    fish_nonlinear = simulate(fish_study, coefficients, fish_w, nonlinear=True)
    fish_record = {
        "case": "high_speed_payload_fishhook",
        "source_output": "constructed amplitude/rate-feasible fishhook",
        "point": {k: fish_point[k] for k in (
            "set", "speed_kmh", "mass_fraction", "payload_height"
        )},
        "linear": summarize_trajectory(fish_linear),
        "nonlinear": summarize_trajectory(fish_nonlinear),
    }
    fish_record["front_peak_change_percent"] = 100.0 * (
        fish_record["nonlinear"]["peak_front_LTR"] /
        max(fish_record["linear"]["peak_front_LTR"], 1e-30) - 1.0
    )
    fish_record["rear_peak_change_percent"] = 100.0 * (
        fish_record["nonlinear"]["peak_rear_LTR"] /
        max(fish_record["linear"]["peak_rear_LTR"], 1e-30) - 1.0
    )
    np.savez_compressed(
        OUT / "high_speed_payload_fishhook.npz",
        steering_witness=fish_w,
        time=fish_nonlinear["time"],
        steering=fish_nonlinear["steering"],
        linear_state=fish_linear["state"],
        nonlinear_state=fish_nonlinear["state"],
        linear_current=fish_linear["current"],
        nonlinear_current=fish_nonlinear["current"],
        linear_outputs=fish_linear["outputs"],
        nonlinear_outputs=fish_nonlinear["outputs"],
    )
    records.append(fish_record)

    summary = {
        "date": "2026-09-23",
        "status": "PASS_SCREEN" if all(
            row["nonlinear"]["finite"] and
            not row["nonlinear"]["wheel_lift_proxy"]
            for row in records
        ) else "REVIEW",
        "case_count": len(records),
        "jacobian_audits": jacobians,
        "maximum_jacobian_relative_error": max(
            max(row["relative_A"], row["relative_B"], row["relative_D"])
            for row in jacobians
        ),
        "maximum_nonlinear_front_LTR": max(
            row["nonlinear"]["peak_front_LTR"] for row in records
        ),
        "maximum_nonlinear_rear_LTR": max(
            row["nonlinear"]["peak_rear_LTR"] for row in records
        ),
        "maximum_nonlinear_current_A": max(
            row["nonlinear"]["peak_current_A"] for row in records
        ),
        "maximum_absolute_front_change_percent": max(
            abs(row["front_peak_change_percent"]) for row in records
        ),
        "maximum_absolute_rear_change_percent": max(
            abs(row["rear_peak_change_percent"]) for row in records
        ),
        "scope": (
            "Critical-case simulation on a Jacobian-matched nonlinear screening "
            "surrogate; not a nonlinear certificate, wheel-lift proof or experiment."
        ),
        "model_features": [
            "load-transfer-sensitive smooth tyre saturation",
            "20 mA current saturation",
            "smooth spool-flow saturation",
            "pressure relief above 9.5 MPa",
            "cubic suspension stops beyond 5 degrees",
        ],
        "source_hashes": {
            "run_nonlinear.py": sha256(Path(__file__)),
            "nonlinear_screen.py": sha256(HERE / "nonlinear_screen.py"),
            "delay_aware_coefficients.csv": sha256(COEFFICIENTS),
            "linear_results.json": sha256(RESULTS),
        },
    }
    base.save_json(OUT / "results.json", {"summary": summary, "cases": records})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()