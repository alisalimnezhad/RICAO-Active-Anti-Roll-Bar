"""Independent reconstruction of the five distinct limiting records.

This verifier does not import the v14 audit wrapper.  It reconstructs the
mapping directly on top of the validated stage-4 model.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
V4 = ROOT / "certified_aarb_v4" / "study.py"
_spec = importlib.util.spec_from_file_location("independent_stage4", V4)
v4 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = v4
_spec.loader.exec_module(v4)

FIELDS = (
    "Cf", "Cr", "kf", "kr", "bf", "br", "ktf", "ktr",
    "area", "Vt", "bulk", "Kx", "KP", "tau", "Kv",
)
TOL = 1e-10


def rebuild(unit, radius):
    unit = np.asarray(unit, dtype=float)
    speed = 50.0 + 25.0 * unit[0]
    fraction = 0.10 * unit[1]
    height = v4.p.h + (1.80 - v4.p.h) * unit[2]
    overrides = v4.distributed_payload_overrides(fraction, height)
    factors = 1.0 + float(radius) * (2.0 * unit[3:] - 1.0)
    for field, factor in zip(FIELDS, factors[:15]):
        overrides[field] = float(getattr(v4.p, field) * factor)
    nominal_gain = v4.ParametricStudy(speed_kmh=70.0).K.copy()
    nominal_fir = np.loadtxt(
        ROOT / "certified_aarb_v5_rear_audit" / "outputs" /
        "reserve_5pct" / "coefficients.csv", delimiter=",",
    ).reshape(2, 6)
    gain = nominal_gain * factors[15:35].reshape(2, 10)
    fir = nominal_fir * factors[35:47].reshape(2, 6)
    proposed = v4.ParametricStudy(
        speed_kmh=speed, overrides=overrides, feedback_gain=gain, phases=9
    )
    baseline = v4.ParametricStudy(
        speed_kmh=speed, overrides=overrides,
        feedback_gain=nominal_gain, phases=9,
    )
    return proposed, baseline, fir


def all_outputs(study, fir, phases=41):
    result = {"spectral_radius": float(study.rho_closed)}
    if study.rho_closed >= 1.0:
        return result
    evaluation = study.evaluate(fir.reshape(-1), phases=phases)
    values = np.asarray(evaluation["continuous_upper"])
    result.update({
        "worst_ltr": float(values[:2].max()),
        "max_utilization": float(values[2:].max()),
    })
    return result


def rear_margin(proposed, baseline, fir, phases=81):
    times = np.linspace(0.0, proposed.h, phases)
    proposed_g = proposed.combine_maps(proposed.maps(times), fir.reshape(-1))
    baseline_g = baseline.combine_maps(baseline.maps(times), np.zeros(12))
    proposed_upper = []
    baseline_lower = []
    for phase in range(phases):
        upper, _, _, _ = proposed.support(
            proposed_g[phase, 1], return_witness=True
        )
        _, _, feasible, _ = baseline.support(
            baseline_g[phase, 1], return_witness=True
        )
        proposed_upper.append(float(upper))
        baseline_lower.append(float(feasible))
    remainder, _, _ = proposed._intersample_correction(
        fir.reshape(-1), proposed.h / (phases - 1)
    )
    upper = max(proposed_upper) + float(remainder[1])
    lower = max(baseline_lower)
    return float((lower - upper) / lower)


def close(left, right):
    return abs(float(left) - float(right)) <= TOL


def main():
    full = json.loads((OUT / "results.json").read_text())
    refined = json.loads((OUT / "critical_refinement.json").read_text())
    source = {record["id"]: record for record in full["records"]}
    identifiers = sorted({
        identifier
        for bank in refined.values()
        for identifier in bank["identifiers"].values()
    })
    records = []
    for identifier in identifiers:
        original = source[identifier]
        proposed, baseline, fir = rebuild(original["unit"], original["radius"])
        computed = all_outputs(proposed, fir)
        if proposed.rho_closed < 1.0 and baseline.rho_closed < 1.0:
            computed["relative_rear_margin"] = rear_margin(
                proposed, baseline, fir
            )
        stored_records = [
            record
            for bank in refined.values()
            for record in bank["records"].values()
            if record["id"] == identifier
        ]
        comparisons = []
        for stored in stored_records:
            checks = {
                "spectral_radius": close(
                    computed["spectral_radius"], stored["spectral_radius"]
                )
            }
            if stored["proposed_stable"]:
                checks.update({
                    "worst_ltr": close(computed["worst_ltr"], stored["worst_ltr"]),
                    "max_utilization": close(
                        computed["max_utilization"], stored["max_utilization"]
                    ),
                    "relative_rear_margin": close(
                        computed["relative_rear_margin"],
                        stored["rear_certificate"]["relative_margin"],
                    ),
                })
            comparisons.append(checks)
        records.append({
            "id": identifier,
            "computed": computed,
            "comparisons": comparisons,
            "pass": all(all(checks.values()) for checks in comparisons),
        })
    result = {
        "date": "2026-09-27",
        "tolerance": TOL,
        "record_count": len(records),
        "records": records,
        "status": "PASS" if len(records) == 5 and all(r["pass"] for r in records)
        else "FAIL",
    }
    (OUT / "independent_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()