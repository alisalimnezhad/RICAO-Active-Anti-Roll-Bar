"""Run the certified fairness, Pareto and finite-model-bank study."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import numpy as np

from study import (
    ParametricStudy, RobustDesigner, base_caps, central_payload_overrides,
    proportional_mass_overrides, evaluate_bank, base,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def compact_evaluation(studies, f, phases=41):
    rows = []
    for row in evaluate_bank(studies, f, phases=phases):
        rows.append({
            "label": row["label"],
            "speed_kmh": row["speed_kmh"],
            "spectral_radius": row["spectral_radius"],
            "continuous": dict(zip(studies[0].names, row["continuous"])),
            "grid": dict(zip(studies[0].names, row["grid"])),
            "interpolation": dict(zip(studies[0].names, row["interpolation"])),
        })
    return rows


def max_named(rows, name):
    return max(float(row["continuous"][name]) for row in rows)


def run_design(name, studies, objective_outputs, caps, candidates,
               tolerance=4e-6):
    saved_path = OUT / f"{name}.json"
    if saved_path.exists():
        print(f"\n=== {name}: reusing completed checkpoint ===", flush=True)
        return json.loads(saved_path.read_text())
    print(f"\n=== {name} ===", flush=True)
    start = time.time()
    designer = RobustDesigner(studies, objective_outputs=objective_outputs, caps=caps)
    result = designer.synthesize(candidates, tolerance=tolerance, verbose=True)
    evaluated = compact_evaluation(studies, result.f, phases=41)
    record = {
        "name": name,
        "objective_outputs": list(objective_outputs),
        "coefficients": result.f,
        "design_gamma": result.gamma,
        "scenario_lower": result.lower,
        "scenario_count": len(result.scenarios),
        "iterations": len(result.history),
        "final_separation_violation": result.history[-1]["max_separation_violation"],
        "evaluated": evaluated,
        "worst_continuous_front_LTR": max_named(evaluated, "front_LTR"),
        "worst_continuous_rear_LTR": max_named(evaluated, "rear_LTR"),
        "worst_continuous_LTR": max(
            max_named(evaluated, "front_LTR"), max_named(evaluated, "rear_LTR")
        ),
        "worst_front_current_mA": 20.0 * max_named(evaluated, "front_current"),
        "worst_rear_current_mA": 20.0 * max_named(evaluated, "rear_current"),
        "worst_constraint_utilization": max(
            max(float(v) for key, v in row["continuous"].items()
                if key not in ("front_LTR", "rear_LTR"))
            for row in evaluated
        ),
        "runtime_seconds": time.time() - start,
        "history": result.history,
    }
    base.save_json(OUT / f"{name}.json", record)
    np.savetxt(OUT / f"{name}_coefficients.csv", result.f.reshape(2, 6), delimiter=",")
    print(json.dumps(base.serial({k: record[k] for k in (
        "design_gamma", "scenario_lower", "iterations",
        "worst_continuous_front_LTR", "worst_continuous_rear_LTR",
        "worst_front_current_mA", "worst_rear_current_mA",
        "worst_constraint_utilization", "runtime_seconds")}), indent=2), flush=True)
    return record


def model_bank(feedback_by_speed):
    loads = [
        ("nominal", {}),
        ("uniform-mass-plus-10pct", proportional_mass_overrides(0.10)),
        ("central-high-payload-plus-10pct", central_payload_overrides(0.10, 1.80)),
    ]
    studies = []
    for speed in (50.0, 70.0, 75.0):
        gain = feedback_by_speed[speed]
        for load_name, overrides in loads:
            studies.append(ParametricStudy(
                speed_kmh=speed, overrides=overrides, feedback_gain=gain,
                phases=9, label=f"{speed:g} km/h; {load_name}",
            ))
    return studies, loads


def source_hashes():
    files = [
        ROOT / "validation" / "parameters.py",
        ROOT / "validation" / "vehicle_model.py",
        ROOT / "validation" / "lqr_controller.py",
        ROOT / "certified_aarb_v2" / "study.py",
        HERE / "study.py",
        Path(__file__),
    ]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def main():
    total_start = time.time()
    previous = json.loads((ROOT / "certified_aarb_v2" / "outputs" /
                           "reproduction_20mA.json").read_text())
    f_v2 = np.asarray(previous["controller_coefficients"], dtype=float).reshape(-1)
    zero = np.zeros(12)

    nominal = ParametricStudy(speed_kmh=70.0, phases=9, label="70 km/h; nominal")
    k70 = nominal.K.copy()
    baseline_eval = compact_evaluation([nominal], zero)[0]
    baseline = baseline_eval["continuous"]
    baseline_grid = baseline_eval["grid"]
    protocol = {
        "date": "2026-09-17",
        "steering_family": {"amplitude_deg": 2.0, "rate_deg_s": 5.0,
                            "horizon_s": 4.0, "sample_s": 0.02,
                            "hold": "FOH steering; ZOH current"},
        "nominal_baseline": baseline_eval,
        "pareto_rear_targets": [float(baseline["rear_LTR"]), 0.028, 0.036, 0.046, 0.060],
        "fair_current_caps_mA": [20.0 * baseline["front_current"],
                                 20.0 * baseline["rear_current"]],
        "finite_bank_speeds_kmh": [50.0, 70.0, 75.0],
        "load_cases": {
            "nominal": {},
            "uniform-mass-plus-10pct": proportional_mass_overrides(0.10),
            "central-high-payload-plus-10pct": central_payload_overrides(0.10, 1.80),
        },
        "load_interpretation": (
            "Analytical stress tests, not identified payload data. The central point "
            "payload is at the nominal longitudinal CG and 1.80 m above the roll axis; "
            "mass, total mass, sprung CG height and roll inertia are coupled."
        ),
        "continuous_parameter_claim": False,
        "source_hashes_before": source_hashes(),
    }
    base.save_json(OUT / "protocol.json", protocol)

    # 1. Certified nominal Pareto frontier at the full 20 mA authority.
    pareto = []
    warm = f_v2.copy()
    for index, rear_target in enumerate(protocol["pareto_rear_targets"]):
        caps = base_caps([nominal], current_caps=[0.998, 0.998], other_margin=0.994)
        # Reserve an explicit continuous-time guard.  The final 41-phase
        # certificate, rather than this grid cap, is the reported quantity.
        guard = max(0.00024, float(baseline_eval["interpolation"]["rear_LTR"]) + 2e-5)
        caps[0, 1] = rear_target - guard
        record = run_design(
            f"pareto_{index}", [nominal], (0,), caps,
            [zero, f_v2, warm], tolerance=4e-6,
        )
        record["rear_target"] = rear_target
        record["rear_design_cap"] = float(caps[0, 1])
        base.save_json(OUT / f"pareto_{index}.json", record)
        pareto.append(record)
        warm = np.asarray(record["coefficients"])
        base.save_json(OUT / "pareto_partial.json", pareto)

    # 2. Equal-actual-peak-current authority, first balanced and then with the
    # rear axle protected at the LQR1 certificate.
    exact_current_caps = np.array([
        baseline["front_current"] + 2e-6,
        baseline["rear_current"] + 2e-6,
    ])
    caps = base_caps([nominal], current_caps=exact_current_caps, other_margin=0.994)
    fair_balanced = run_design(
        "fair_current_balanced", [nominal], (0, 1), caps,
        [zero, f_v2, pareto[0]["coefficients"]], tolerance=4e-6,
    )
    caps_protected = caps.copy()
    caps_protected[0, 1] = baseline_grid["rear_LTR"] - 1e-5
    fair_protected = run_design(
        "fair_current_rear_protected", [nominal], (0,), caps_protected,
        [zero, fair_balanced["coefficients"], pareto[0]["coefficients"]],
        tolerance=4e-6,
    )

    # 3. Finite multi-model bank.  First screen 80 and 90 km/h under the
    # declared 3-deg tyre-slip guard; these are not silently included if the
    # analysis domain is already violated by the reference controller.
    domain_screen = []
    for speed in (75.0, 80.0, 90.0):
        for load_name, overrides in [
            ("nominal", {}),
            ("uniform-mass-plus-10pct", proportional_mass_overrides(0.10)),
            ("central-high-payload-plus-10pct", central_payload_overrides(0.10, 1.80)),
        ]:
            study = ParametricStudy(speed_kmh=speed, overrides=overrides,
                                    feedback_gain=k70, phases=9,
                                    label=f"{speed:g} km/h; {load_name}")
            row = compact_evaluation([study], zero, phases=41)[0]
            row["maximum_non_LTR_utilization"] = max(
                float(v) for key, v in row["continuous"].items()
                if key not in ("front_LTR", "rear_LTR")
            )
            domain_screen.append(row)
    base.save_json(OUT / "domain_screen.json", domain_screen)

    fixed_bank, load_defs = model_bank({50.0: k70, 70.0: k70, 75.0: k70})
    fixed_baseline_rows = compact_evaluation(fixed_bank, zero)
    fixed_v2_rows = compact_evaluation(fixed_bank, f_v2)
    bank_caps = base_caps(fixed_bank, current_caps=[0.994, 0.994], other_margin=0.990)
    fixed_balanced = run_design(
        "bank_fixed_balanced", fixed_bank, (0, 1), bank_caps,
        [zero, f_v2, fair_balanced["coefficients"]], tolerance=7e-6,
    )
    protected_caps = bank_caps.copy()
    for i, row in enumerate(fixed_baseline_rows):
        protected_caps[i, 1] = float(row["grid"]["rear_LTR"]) - 6e-5
    fixed_protected = run_design(
        "bank_fixed_rear_protected", fixed_bank, (0,), protected_caps,
        [zero, fixed_balanced["coefficients"], fair_protected["coefficients"]],
        tolerance=7e-6,
    )

    # 4. Speed-scheduled feedback and FIR coefficients, while remaining robust
    # to all three declared load cases at each measured speed.
    scheduled = []
    schedule_gains = {}
    for speed in (50.0, 70.0, 75.0):
        schedule_gains[speed] = ParametricStudy(speed_kmh=speed, phases=9).K.copy()
    scheduled_bank, _ = model_bank(schedule_gains)
    for speed in (50.0, 70.0, 75.0):
        group = [study for study in scheduled_bank if study.speed_kmh == speed]
        group_caps = base_caps(group, current_caps=[0.994, 0.994], other_margin=0.990)
        candidates = [zero, f_v2, fixed_balanced["coefficients"]]
        record = run_design(
            f"bank_scheduled_{str(speed).replace('.', 'p')}", group, (0, 1),
            group_caps, candidates, tolerance=7e-6,
        )
        scheduled.append(record)

    source_after = source_hashes()
    if source_after != protocol["source_hashes_before"]:
        raise AssertionError("source files changed during the numerical run")

    summary = {
        "status": "PASS",
        "protocol": protocol,
        "nominal_pareto": pareto,
        "fair_current_balanced": fair_balanced,
        "fair_current_rear_protected": fair_protected,
        "domain_screen": domain_screen,
        "fixed_bank_baseline": fixed_baseline_rows,
        "fixed_bank_v2_controller": fixed_v2_rows,
        "fixed_bank_balanced": fixed_balanced,
        "fixed_bank_rear_protected": fixed_protected,
        "speed_scheduled": scheduled,
        "source_hashes_after": source_after,
        "total_runtime_seconds": time.time() - total_start,
    }
    base.save_json(OUT / "stage3_results.json", summary)
    print("\nSTAGE 3 COMPLETE", flush=True)
    print(json.dumps({
        "fair_balanced": [fair_balanced["worst_continuous_front_LTR"],
                          fair_balanced["worst_continuous_rear_LTR"]],
        "fair_protected": [fair_protected["worst_continuous_front_LTR"],
                           fair_protected["worst_continuous_rear_LTR"]],
        "fixed_bank": [fixed_balanced["worst_continuous_front_LTR"],
                       fixed_balanced["worst_continuous_rear_LTR"]],
        "fixed_protected": [fixed_protected["worst_continuous_front_LTR"],
                            fixed_protected["worst_continuous_rear_LTR"]],
        "scheduled_worst": max(r["worst_continuous_LTR"] for r in scheduled),
        "runtime_seconds": summary["total_runtime_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()