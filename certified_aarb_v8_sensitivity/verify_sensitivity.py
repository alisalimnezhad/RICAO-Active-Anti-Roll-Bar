"""Independent consistency checks for the version-8 sensitivity package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(left, right, tol=2e-12):
    return abs(float(left) - float(right)) <= tol


def main():
    data = json.loads((OUT / "results.json").read_text())
    cases = {case["key"]: case for case in data["cases"]}
    groups = {name: [case for case in data["cases"] if case["group"] == name]
              for name in ("fir", "horizon", "steering", "hydraulic")}
    baseline_keys = ("fir_6", "horizon_4s", "steering_2deg_5dps")
    baseline_fields = (
        "worst_front_LTR", "worst_rear_LTR", "worst_LTR",
        "max_constraint_utilization", "minimum_rear_margin_percent",
        "max_spectral_radius",
    )
    baseline_consistency = {
        field: max(float(cases[key][field]) for key in baseline_keys)
        - min(float(cases[key][field]) for key in baseline_keys)
        for field in baseline_fields
    }
    source_hashes = data["source_hashes"]
    source_checks = {
        "run_sensitivity.py": source_hashes["run_sensitivity.py"] == sha256(HERE / "run_sensitivity.py"),
        "study.py": source_hashes["study.py"] == sha256(HERE / "study.py"),
        "selected_coefficients": source_hashes["selected_coefficients"] == sha256(
            ROOT / "certified_aarb_v5_rear_audit/outputs/comparison/selected_reserve_5pct_coefficients.csv"
        ),
        "v4_study.py": source_hashes["v4_study.py"] == sha256(ROOT / "certified_aarb_v4/study.py"),
        "vehicle_model.py": source_hashes["vehicle_model.py"] == sha256(ROOT / "validation/vehicle_model.py"),
        "parameters.py": source_hashes["parameters.py"] == sha256(ROOT / "validation/parameters.py"),
    }
    delay = {row["delay_ms"]: row for row in data["delay"]}
    old5 = json.loads((
        ROOT / "certified_aarb_v5_rear_audit/outputs/delay_sensitivity/delay_5ms/results.json"
    ).read_text())["summary"]["stability"]
    old10 = json.loads((
        ROOT / "certified_aarb_v5_rear_audit/outputs/delay_sensitivity/delay_10ms/results.json"
    ).read_text())["summary"]["stability"]
    checks = {
        "results_complete": data["status"] == "COMPLETE",
        "case_count_48": len(cases) == 48,
        "group_counts": [len(groups[name]) for name in ("fir", "horizon", "steering", "hydraulic")] == [5, 3, 9, 31],
        "fixed_controller_only": all(not case["redesign_performed"] for case in cases.values()),
        "baseline_cases_consistent": max(baseline_consistency.values()) <= 2e-12,
        "four_and_six_tap_pass": cases["fir_4"]["status"] == cases["fir_6"]["status"] == "PASS",
        "short_fir_failures_retained": all(cases[key]["status"] != "PASS" for key in ("fir_1", "fir_2", "fir_3")),
        "horizon_cases_pass": all(case["status"] == "PASS" for case in groups["horizon"]),
        "horizon_front_converged": max(c["worst_front_LTR"] for c in groups["horizon"]) - min(c["worst_front_LTR"] for c in groups["horizon"]) < 5e-7,
        "nominal_steering_pass": cases["steering_2deg_5dps"]["status"] == "PASS",
        "zero_delay_identity": delay[0.0]["status"] == "PASS" and delay[0.0]["stable_count"] == 170,
        "nonzero_delays_unstable": all(delay[value]["status"] == "FAIL_UNSTABLE" for value in (5.0, 10.0, 20.0)),
        "delay_5_matches_archive": close(delay[5.0]["maximum_spectral_radius"], old5["maximum_spectral_radius"]),
        "delay_10_matches_archive": close(delay[10.0]["maximum_spectral_radius"], old10["maximum_spectral_radius"]),
        "hydraulic_instabilities_retained": sum(case["status"] == "FAIL_UNSTABLE" for case in groups["hydraulic"]) >= 1,
        "source_hashes_match": all(source_checks.values()),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    audit = {
        "status": status,
        "date": "2026-09-23",
        "checks": checks,
        "source_hash_checks": source_checks,
        "baseline_consistency": baseline_consistency,
        "headline": {
            "minimum_passing_taps": 4,
            "horizon_front_range": max(c["worst_front_LTR"] for c in groups["horizon"]) - min(c["worst_front_LTR"] for c in groups["horizon"]),
            "delay_20ms_maximum_spectral_radius": delay[20.0]["maximum_spectral_radius"],
            "unstable_hydraulic_case_count": sum(case["status"] == "FAIL_UNSTABLE" for case in groups["hydraulic"]),
        },
        "scope": data["protocol"],
        "limitations": data["limitations"],
    }
    (OUT / "verification.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = {}
    for path in sorted(item for item in OUT.rglob("*")
                       if item.is_file() and item.name != "artifact_manifest.json"):
        manifest[str(path.relative_to(HERE))] = sha256(path)
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, **audit["headline"]}, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()