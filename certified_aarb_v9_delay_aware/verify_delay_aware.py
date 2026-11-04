"""Independent consistency checks for the delay-aware and nonlinear artifacts."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from delay_design import base, delayed_study


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    gain = json.loads((OUT / "gain_scan.json").read_text())
    design = json.loads((OUT / "design.json").read_text())
    result = json.loads((OUT / "results.json").read_text())
    nonlinear = json.loads((OUT / "nonlinear" / "results.json").read_text())
    coefficients = np.loadtxt(OUT / "coefficients.csv", delimiter=",").reshape(-1)
    summary = result["summary"]
    selected = min(gain["rows"], key=lambda row: row["maximum_spectral_radius"])
    checks = {
        "gain_grid_minimum_selected": bool(abs(
            float(selected["scale"]) - float(gain["selected_scale"])
        ) < 1e-14),
        "design_coefficients_match": bool(np.max(np.abs(
            coefficients - np.asarray(design["coefficients"])
        )) < 1e-14),
        "verification_has_170_points": len(result["rows"]) == 170,
        "all_170_stable": summary["stable_count"] == 170,
        "all_170_constraint_feasible": summary["constraint_feasible_count"] == 170,
        "all_170_rear_non_degrading": summary["rear_non_degrading_count"] == 170,
        "nonlinear_five_cases": nonlinear["summary"]["case_count"] == 5,
        "jacobian_gate_passed": nonlinear["summary"]["maximum_jacobian_relative_error"] < 2e-5,
        "nonlinear_screen_finite": all(
            row["nonlinear"]["finite"] for row in nonlinear["cases"]
        ),
        "nonlinear_source_hashes_match": all((
            nonlinear["summary"]["source_hashes"]["run_nonlinear.py"]
            == sha256(HERE / "run_nonlinear.py"),
            nonlinear["summary"]["source_hashes"]["nonlinear_screen.py"]
            == sha256(HERE / "nonlinear_screen.py"),
            nonlinear["summary"]["source_hashes"]["delay_aware_coefficients.csv"]
            == sha256(OUT / "coefficients.csv"),
            nonlinear["summary"]["source_hashes"]["linear_results.json"]
            == sha256(OUT / "results.json"),
        )),
    }

    point = {"speed_kmh": 70.0, "mass_fraction": 0.0,
             "payload_height": 1.15, "set": "verification"}
    nominal_gain = np.asarray(gain["nominal_gain"])
    study = delayed_study(
        point, float(gain["selected_scale"]) * nominal_gain, phases=5
    )
    rng = np.random.default_rng(20260923)
    increments = rng.uniform(-base.RHO, base.RHO, size=study.n)
    witness = np.clip(np.cumsum(increments), -1.0, 1.0)
    # Projection back through rate clipping keeps the independent witness in W.
    for k in range(1, study.n):
        witness[k] = np.clip(witness[k], witness[k-1]-base.RHO,
                             witness[k-1]+base.RHO)
    phases = np.linspace(0.0, study.h, 7)
    direct = study.direct_final_interval(coefficients, witness, phases)
    maps = study.combine_maps(study.maps(phases), coefficients)
    lifted = np.einsum("pin,n->pi", maps, witness)
    checks["direct_recurrence_matches_lift"] = bool(
        float(np.max(np.abs(direct - lifted))) < 2e-10
    )
    payload = {
        "date": "2026-09-23",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "headline": {
            "feedback_gain_scale": summary["feedback_gain_scale"],
            "maximum_spectral_radius": summary["maximum_spectral_radius"],
            "worst_LTR": summary["worst_LTR"],
            "maximum_constraint_utilization": summary["max_constraint_utilization"],
            "maximum_nonlinear_rear_LTR": nonlinear["summary"]["maximum_nonlinear_rear_LTR"],
        },
    }
    base.save_json(OUT / "verification.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()