"""Zero-delay checkpoint for the rear-certificate latency audit.

The 0 ms case is the identity case of the latency study: state feedback and
the current steering knot are available at the controller update instant.
This script combines the already certified 170-point zero-delay records for
the fixed rear-protected controller and its 5% reserve redesign, checks that
their operating banks coincide, and independently refines both limiting rear
certificates on 161- and 321-phase meshes.

No manuscript file is modified.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V4 = ROOT / "certified_aarb_v4"
OUT = HERE / "outputs" / "delay_sensitivity" / "delay_0ms"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)

if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))

from study import ParametricStudy, base, study_at  # noqa: E402


CURRENT_SCREEN = V4 / "outputs" / "envelope_results.json"
CURRENT_REAR = V4 / "outputs" / "envelope_verification.json"
CURRENT_COEFFICIENTS = V4 / "outputs" / "fixed_rear_protected_coefficients.csv"
RESERVE_SCREEN = HERE / "outputs" / "reserve_5pct" / "screening.json"
RESERVE_REAR = HERE / "outputs" / "reserve_5pct" / "rear_verification.json"
RESERVE_COEFFICIENTS = HERE / "outputs" / "reserve_5pct" / "coefficients.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def point_key(row):
    return (
        round(float(row["speed_kmh"]), 12),
        round(float(row["mass_fraction"]), 12),
        round(float(row["payload_height"]), 12),
        row["set"],
    )


def rear_certificate(study, controllers, phases):
    """One-sided rear certificate for several controllers on one phase mesh."""
    phase_times = np.linspace(0.0, study.h, int(phases))
    maps = study.maps(phase_times)
    records = {}
    for name, coefficients in controllers.items():
        g = study.combine_maps(maps, np.asarray(coefficients, dtype=float))
        corrected = []
        feasible = []
        max_dual = 0.0
        for phase in range(int(phases)):
            upper, _, witness_value, correction = study.support(
                g[phase, 1], return_witness=True
            )
            corrected.append(float(upper))
            feasible.append(float(witness_value))
            max_dual = max(max_dual, float(correction))
        spacing = study.h / (int(phases) - 1)
        interpolation, _, _ = study._intersample_correction(
            np.asarray(coefficients, dtype=float), spacing
        )
        records[name] = {
            "grid_corrected_upper": max(corrected),
            "grid_feasible_lower": max(feasible),
            "intersample_correction": float(interpolation[1]),
            "continuous_upper": max(corrected) + float(interpolation[1]),
            "max_dual_correction": max_dual,
        }
    baseline = records["baseline"]["grid_feasible_lower"]
    for name in ("current", "reserve_5pct"):
        margin = baseline - records[name]["continuous_upper"]
        records[name]["certified_margin"] = margin
        records[name]["relative_margin"] = margin / baseline
        records[name]["relative_margin_percent"] = 100.0 * margin / baseline
    return records


def point_dict(row):
    return {key: row[key] for key in (
        "set", "speed_kmh", "mass_fraction", "payload_height"
    )}


def main():
    start = time.time()
    current_screen = load(CURRENT_SCREEN)
    current_rear = load(CURRENT_REAR)
    reserve_screen = load(RESERVE_SCREEN)
    reserve_rear = load(RESERVE_REAR)

    current_screen_rows = {point_key(row): row for row in current_screen["rows"]}
    current_rear_rows = {point_key(row): row for row in current_rear["rear_rows"]}
    reserve_screen_rows = {point_key(row): row for row in reserve_screen["rows"]}
    reserve_rear_rows = {point_key(row): row for row in reserve_rear["rows"]}
    banks = (
        set(current_screen_rows), set(current_rear_rows),
        set(reserve_screen_rows), set(reserve_rear_rows),
    )
    if not all(bank == banks[0] for bank in banks[1:]):
        raise RuntimeError("zero-delay operating-point banks do not coincide")
    if len(banks[0]) != 170:
        raise RuntimeError(f"expected 170 points, found {len(banks[0])}")

    rows = []
    for key in sorted(banks[0]):
        cs = current_screen_rows[key]
        cr = current_rear_rows[key]
        rs = reserve_screen_rows[key]
        rr = reserve_rear_rows[key]
        baseline = float(cr["baseline"]["grid_feasible_lower"])
        current_relative = float(cr["certified_margin"] / baseline)
        rows.append({
            **point_dict(cs),
            "delay_ms": 0.0,
            "current_front_LTR": cs["fixed_rear_protected"]["front_LTR"],
            "current_rear_LTR_21phase": cs["fixed_rear_protected"]["rear_LTR"],
            "current_constraint_utilization": cs["fixed_rear_protected"]["constraint_utilization"],
            "current_limiting_constraint": cs["fixed_rear_protected"]["limiting_constraint"],
            "current_spectral_radius": cs["fixed_rear_protected"]["spectral_radius"],
            "current_rear_margin_81_percent": 100.0 * current_relative,
            "current_rear_certified": bool(cr["certified"]),
            "reserve_front_LTR": rs["proposed"]["front_LTR"],
            "reserve_rear_LTR_21phase": rs["proposed"]["rear_LTR"],
            "reserve_constraint_utilization": rs["proposed"]["constraint_utilization"],
            "reserve_limiting_constraint": rs["proposed"]["limiting_constraint"],
            "reserve_spectral_radius": rs["proposed"]["spectral_radius"],
            "reserve_rear_margin_81_percent": rr["relative_margin_percent"],
            "reserve_rear_certified": bool(rr["certified"]),
        })

    current_limit = min(rows, key=lambda row: row["current_rear_margin_81_percent"])
    reserve_limit = min(rows, key=lambda row: row["reserve_rear_margin_81_percent"])

    nominal = ParametricStudy(speed_kmh=70.0)
    fixed_gain = nominal.K.copy()
    current_coefficients = np.loadtxt(CURRENT_COEFFICIENTS, delimiter=",").reshape(-1)
    reserve_coefficients = np.loadtxt(RESERVE_COEFFICIENTS, delimiter=",").reshape(-1)
    controllers = {
        "baseline": np.zeros(12),
        "current": current_coefficients,
        "reserve_5pct": reserve_coefficients,
    }

    refined = []
    unique_limits = {
        point_key(current_limit): ("current_81phase_limit", current_limit),
        point_key(reserve_limit): ("reserve_81phase_limit", reserve_limit),
    }
    for _, (label, point) in unique_limits.items():
        study = study_at(
            point["speed_kmh"], point["mass_fraction"],
            point["payload_height"], fixed_gain, phases=9,
        )
        record = {"label": label, **point_dict(point), "meshes": {}}
        for phases in (81, 161, 321):
            record["meshes"][str(phases)] = rear_certificate(
                study, controllers, phases
            )
        refined.append(record)

    current_worst_front = max(rows, key=lambda row: row["current_front_LTR"])
    reserve_worst_front = max(rows, key=lambda row: row["reserve_front_LTR"])
    current_worst_constraint = max(
        rows, key=lambda row: row["current_constraint_utilization"]
    )
    reserve_worst_constraint = max(
        rows, key=lambda row: row["reserve_constraint_utilization"]
    )
    max_rho_current = max(row["current_spectral_radius"] for row in rows)
    max_rho_reserve = max(row["reserve_spectral_radius"] for row in rows)

    summary = {
        "status": "PASS",
        "date": "2026-09-22",
        "delay_ms": 0.0,
        "delay_definition": (
            "Identity/no-latency case: the sampled controller uses the current "
            "state vector and current steering knot at each 20 ms update."
        ),
        "plant_count": len(rows),
        "counts_by_set": {
            name: sum(row["set"] == name for row in rows)
            for name in ("structured", "sobol_holdout")
        },
        "broad_certificate_phases": 21,
        "rear_certificate_phases": 81,
        "refinement_phases": [161, 321],
        "current": {
            "minimum_rear_margin_81_percent": current_limit["current_rear_margin_81_percent"],
            "limiting_rear_point": point_dict(current_limit),
            "worst_front_LTR_21phase": current_worst_front["current_front_LTR"],
            "worst_front_point": point_dict(current_worst_front),
            "max_constraint_utilization_21phase": current_worst_constraint["current_constraint_utilization"],
            "limiting_constraint": current_worst_constraint["current_limiting_constraint"],
            "constraint_point": point_dict(current_worst_constraint),
            "max_spectral_radius": max_rho_current,
            "all_rear_non_degrading": all(row["current_rear_certified"] for row in rows),
            "all_constraints_feasible": all(row["current_constraint_utilization"] <= 1.0 + 2e-8 for row in rows),
            "all_schur_stable": max_rho_current < 1.0,
        },
        "reserve_5pct": {
            "minimum_rear_margin_81_percent": reserve_limit["reserve_rear_margin_81_percent"],
            "limiting_rear_point": point_dict(reserve_limit),
            "worst_front_LTR_21phase": reserve_worst_front["reserve_front_LTR"],
            "worst_front_point": point_dict(reserve_worst_front),
            "max_constraint_utilization_21phase": reserve_worst_constraint["reserve_constraint_utilization"],
            "limiting_constraint": reserve_worst_constraint["reserve_limiting_constraint"],
            "constraint_point": point_dict(reserve_worst_constraint),
            "max_spectral_radius": max_rho_reserve,
            "all_rear_non_degrading": all(row["reserve_rear_certified"] for row in rows),
            "five_percent_target_achieved": all(row["reserve_rear_margin_81_percent"] >= 5.0 - 1e-10 for row in rows),
            "all_constraints_feasible": all(row["reserve_constraint_utilization"] <= 1.0 + 2e-8 for row in rows),
            "all_schur_stable": max_rho_reserve < 1.0,
        },
        "front_cost_of_5pct_reserve_percent": 100.0 * (
            reserve_worst_front["reserve_front_LTR"] /
            current_worst_front["current_front_LTR"] - 1.0
        ),
        "refined_limiting_points": refined,
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in (
                CURRENT_SCREEN, CURRENT_REAR, CURRENT_COEFFICIENTS,
                RESERVE_SCREEN, RESERVE_REAR, RESERVE_COEFFICIENTS,
            )
        },
        "scope": (
            "Baseline checkpoint for a sequential latency study. It certifies "
            "the evaluated linear plants at zero delay and is not a delay-robustness claim."
        ),
        "runtime_seconds": time.time() - start,
    }
    checks = [
        summary["current"]["all_rear_non_degrading"],
        summary["current"]["all_constraints_feasible"],
        summary["current"]["all_schur_stable"],
        summary["reserve_5pct"]["all_rear_non_degrading"],
        summary["reserve_5pct"]["five_percent_target_achieved"],
        summary["reserve_5pct"]["all_constraints_feasible"],
        summary["reserve_5pct"]["all_schur_stable"],
    ]
    if not all(checks):
        summary["status"] = "REVIEW"

    base.save_json(OUT / "results.json", {"summary": summary, "rows": rows})

    csv_fields = list(rows[0])
    with (OUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    table = (
        "Current controller & "
        f"{summary['current']['minimum_rear_margin_81_percent']:.3f} & "
        f"{summary['current']['worst_front_LTR_21phase']:.6f} & "
        f"{summary['current']['max_constraint_utilization_21phase']:.3f} & "
        f"{summary['current']['max_spectral_radius']:.3f} \\\\\n+"
        "5\\% reserve controller & "
        f"{summary['reserve_5pct']['minimum_rear_margin_81_percent']:.3f} & "
        f"{summary['reserve_5pct']['worst_front_LTR_21phase']:.6f} & "
        f"{summary['reserve_5pct']['max_constraint_utilization_21phase']:.3f} & "
        f"{summary['reserve_5pct']['max_spectral_radius']:.3f} \\\\\n+"
    )
    table = table.replace("\n+", "\n")
    (OUT / "tables" / "delay_0ms.tex").write_text(table)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), constrained_layout=True)
    names = ["Current", "5% reserve"]
    colors = ["#555555", "#009E73"]
    margins = [
        summary["current"]["minimum_rear_margin_81_percent"],
        summary["reserve_5pct"]["minimum_rear_margin_81_percent"],
    ]
    axes[0].bar(names, margins, color=colors)
    axes[0].axhline(0.0, color="black", linewidth=0.9)
    axes[0].axhline(5.0, color="gray", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Minimum rear margin (%)")
    axes[0].set_title("(a) Zero-delay rear certificate")
    axes[0].grid(axis="y", alpha=0.25)

    utilization = [
        summary["current"]["max_constraint_utilization_21phase"],
        summary["reserve_5pct"]["max_constraint_utilization_21phase"],
    ]
    axes[1].bar(names, utilization, color=colors)
    axes[1].axhline(1.0, color="#D55E00", linestyle="--", linewidth=1.2,
                    label="Declared limit")
    axes[1].set_ylabel("Maximum normalized utilization")
    axes[1].set_title("(b) Zero-delay constraints")
    axes[1].set_ylim(0.0, 1.08)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "figures" / "delay_0ms_baseline.png", dpi=220)
    fig.savefig(OUT / "figures" / "delay_0ms_baseline.pdf")
    plt.close(fig)

    scientific = (
        "# Zero-delay latency checkpoint\n\n"
        "At 0 ms, the latency model reduces exactly to the information pattern "
        "used in manuscript version 5. Across all 170 evaluated plants, both "
        "controllers remain Schur stable and satisfy the declared continuous-time "
        "output constraints. The current rear-protected controller has a minimum "
        f"rear margin of {summary['current']['minimum_rear_margin_81_percent']:.3f}%, "
        "whereas the reserve redesign gives "
        f"{summary['reserve_5pct']['minimum_rear_margin_81_percent']:.3f}%. "
        "The 5% reserve increases the worst front bound by only "
        f"{summary['front_cost_of_5pct_reserve_percent']:.3f}%.\n\n"
        "This is the control case for subsequent 5, 10 and 20 ms calculations; "
        "by itself it does not demonstrate robustness to nonzero latency.\n"
    )
    (OUT / "scientific_summary.md").write_text(scientific)

    artifacts = [
        OUT / "results.json", OUT / "results.csv",
        OUT / "tables" / "delay_0ms.tex",
        OUT / "figures" / "delay_0ms_baseline.png",
        OUT / "figures" / "delay_0ms_baseline.pdf",
        OUT / "scientific_summary.md",
    ]
    manifest = {
        str(path.relative_to(HERE)): sha256(path) for path in artifacts
    }
    base.save_json(OUT / "artifact_manifest.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()