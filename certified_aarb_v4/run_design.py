"""Synthesize a fixed rear-protected controller for the stage-4 envelope."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import numpy as np

from study import ParametricStudy, RobustDesigner, base_caps, base, design_skeleton

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.time()
    nominal = ParametricStudy(speed_kmh=70.0)
    k70 = nominal.K.copy()
    studies = design_skeleton(k70, phases=9)
    zero = np.zeros(12)
    old_protected = np.loadtxt(
        ROOT / "certified_aarb_v3/outputs/bank_fixed_rear_protected_coefficients.csv",
        delimiter=",",
    ).reshape(-1)
    old_balanced = np.loadtxt(
        ROOT / "certified_aarb_v3/outputs/bank_fixed_balanced_coefficients.csv",
        delimiter=",",
    ).reshape(-1)

    baselines = []
    caps = base_caps(studies, current_caps=[0.985, 0.985], other_margin=0.980)
    for i, study in enumerate(studies):
        ev = study.evaluate(zero, phases=41)
        baselines.append({
            "label": study.label,
            "continuous": dict(zip(study.names, ev["continuous_upper"])),
            "grid": dict(zip(study.names, ev["grid_upper"])),
        })
        # Reserve more rear-LTR room than version 3 so the independent
        # 81-phase and unseen-parameter checks can remain below LQR1.
        caps[i, 1] = float(ev["grid_upper"][1]) - 1.2e-4

    designer = RobustDesigner(studies, objective_outputs=(0,), caps=caps)
    result = designer.synthesize(
        [zero, old_protected, old_balanced], tolerance=7e-6, verbose=True
    )
    evaluations = []
    for study in studies:
        ev = study.evaluate(result.f, phases=81)
        evaluations.append({
            "label": study.label,
            "continuous": dict(zip(study.names, ev["continuous_upper"])),
            "grid": dict(zip(study.names, ev["grid_upper"])),
            "interpolation": dict(zip(study.names, ev["interpolation_correction"])),
            "spectral_radius": study.rho_closed,
        })

    rear_ok = all(
        row["continuous"]["rear_LTR"] <= baseline["continuous"]["rear_LTR"] + 2e-8
        for row, baseline in zip(evaluations, baselines)
    )
    constraints_ok = all(
        max(v for k, v in row["continuous"].items()
            if k not in ("front_LTR", "rear_LTR")) <= 1.0 + 2e-8
        for row in evaluations
    )
    record = {
        "status": "PASS" if rear_ok and constraints_ok else "REVIEW",
        "date": "2026-09-19",
        "design_models": [study.label for study in studies],
        "coefficients": result.f,
        "design_gamma": result.gamma,
        "scenario_lower": result.lower,
        "iterations": len(result.history),
        "scenario_count": len(result.scenarios),
        "history": result.history,
        "baselines": baselines,
        "evaluations_81_phase": evaluations,
        "checks": {
            "rear_non_degradation_all_design_models": rear_ok,
            "constraints_all_design_models": constraints_ok,
        },
        "worst_front_LTR": max(r["continuous"]["front_LTR"] for r in evaluations),
        "worst_rear_LTR": max(r["continuous"]["rear_LTR"] for r in evaluations),
        "worst_constraint_utilization": max(
            max(v for k, v in r["continuous"].items()
                if k not in ("front_LTR", "rear_LTR"))
            for r in evaluations
        ),
        "source_hashes": {
            "study.py": sha(HERE / "study.py"),
            "run_design.py": sha(Path(__file__)),
            "v3_study.py": sha(ROOT / "certified_aarb_v3/study.py"),
            "vehicle_model.py": sha(ROOT / "validation/vehicle_model.py"),
            "parameters.py": sha(ROOT / "validation/parameters.py"),
        },
        "runtime_seconds": time.time() - start,
    }
    base.save_json(OUT / "fixed_rear_protected_design.json", record)
    np.savetxt(OUT / "fixed_rear_protected_coefficients.csv",
               result.f.reshape(2, 6), delimiter=",")
    print(json.dumps({k: record[k] for k in (
        "status", "design_gamma", "scenario_lower", "iterations",
        "worst_front_LTR", "worst_rear_LTR",
        "worst_constraint_utilization", "checks", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()