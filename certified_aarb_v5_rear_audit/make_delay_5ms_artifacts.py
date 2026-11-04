"""Refine and package the 5 ms latency-audit results."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V4 = ROOT / "certified_aarb_v4"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))

from delay_study import DelayedParametricStudy  # noqa: E402
from study import ParametricStudy, base, distributed_payload_overrides  # noqa: E402


OUT = HERE / "outputs" / "delay_sensitivity" / "delay_5ms"
ZERO = HERE / "outputs" / "delay_sensitivity" / "delay_0ms" / "results.json"
RESULTS = OUT / "results.json"
CURRENT_COEFFICIENTS = V4 / "outputs" / "fixed_rear_protected_coefficients.csv"
RESERVE_COEFFICIENTS = HERE / "outputs" / "reserve_5pct" / "coefficients.csv"
(OUT / "figures").mkdir(exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point_key(row):
    return (
        round(float(row["speed_kmh"]), 12),
        round(float(row["mass_fraction"]), 12),
        round(float(row["payload_height"]), 12),
        row["set"],
    )


def rear_mesh(study, controllers, phases):
    times = np.linspace(0.0, study.h, int(phases))
    maps = study.maps(times)
    records = {}
    for name, coefficients in controllers.items():
        coefficients = np.asarray(coefficients, dtype=float)
        g = study.combine_maps(maps, coefficients)
        corrected, feasible = [], []
        max_correction = 0.0
        for phase in range(int(phases)):
            upper, _, primal, correction = study.support(
                g[phase, 1], return_witness=True
            )
            corrected.append(float(upper))
            feasible.append(float(primal))
            max_correction = max(max_correction, float(correction))
        interpolation, _, _ = study._intersample_correction(
            coefficients, study.h / (int(phases) - 1)
        )
        records[name] = {
            "grid_corrected_upper": max(corrected),
            "grid_feasible_lower": max(feasible),
            "intersample_correction": float(interpolation[1]),
            "continuous_upper": max(corrected) + float(interpolation[1]),
            "max_dual_correction": max_correction,
        }
    baseline = records["baseline"]["grid_feasible_lower"]
    for name in ("current", "reserve_5pct"):
        margin = baseline - records[name]["continuous_upper"]
        records[name]["certified_margin"] = margin
        records[name]["relative_margin_percent"] = 100.0 * margin / baseline
    return records


def main():
    result = json.loads(RESULTS.read_text())
    zero = json.loads(ZERO.read_text())["summary"]
    summary = result["summary"]
    rows = result["rows"]
    current_f = np.loadtxt(CURRENT_COEFFICIENTS, delimiter=",").reshape(-1)
    reserve_f = np.loadtxt(RESERVE_COEFFICIENTS, delimiter=",").reshape(-1)
    controllers = {
        "baseline": np.zeros(12),
        "current": current_f,
        "reserve_5pct": reserve_f,
    }

    nominal = ParametricStudy(speed_kmh=70.0, phases=9)
    fixed_gain = nominal.K.copy()
    limiting_keys = {
        point_key(summary["current"]["limiting_rear_point"]),
        point_key(summary["reserve_5pct"]["limiting_rear_point"]),
    }
    lookup = {point_key(row): row for row in rows}
    refinements = []
    for key in sorted(limiting_keys):
        point = lookup[key]
        study = DelayedParametricStudy(
            speed_kmh=point["speed_kmh"],
            overrides=distributed_payload_overrides(
                point["mass_fraction"], point["payload_height"]
            ),
            feedback_gain=fixed_gain,
            delay_s=0.005,
            phases=9,
        )
        refinements.append({
            "point": {name: point[name] for name in (
                "set", "speed_kmh", "mass_fraction", "payload_height"
            )},
            "spectral_radius": study.rho_closed,
            "meshes": {
                str(phases): rear_mesh(study, controllers, phases)
                for phases in (81, 161, 321)
            },
        })

    # Independent implementation checks for the delayed lifted maps.
    phases = np.linspace(0.0, 0.02, 11)
    delayed_zero = DelayedParametricStudy(
        speed_kmh=70.0, delay_s=0.0, phases=9
    )
    zero_map_error = float(np.max(np.abs(
        nominal.maps(phases) - delayed_zero.maps(phases)
    )))
    zero_rho_error = abs(nominal.rho_closed - delayed_zero.rho_closed)
    rng = np.random.default_rng(20260922)
    increments = rng.uniform(-0.05, 0.05, 200)
    steering = np.clip(np.cumsum(increments), -1.0, 1.0)
    coefficients = rng.uniform(-3.0, 3.0, 12)
    delayed_five = DelayedParametricStudy(
        speed_kmh=70.0, delay_s=0.005, phases=9
    )
    lifted = np.einsum(
        "pin,n->pi",
        delayed_five.combine_maps(delayed_five.maps(phases), coefficients),
        steering,
    )
    direct = delayed_five.direct_final_interval(
        coefficients, steering, phases
    )
    direct_relative_error = float(
        np.max(np.abs(lifted - direct)) /
        max(1.0, float(np.max(np.abs(lifted))), float(np.max(np.abs(direct))))
    )
    validation = {
        "date": "2026-09-22",
        "zero_delay_map_max_abs_error": zero_map_error,
        "zero_delay_spectral_radius_abs_error": zero_rho_error,
        "five_ms_direct_lift_max_relative_error": direct_relative_error,
        "nominal_five_ms_spectral_radius": delayed_five.rho_closed,
        "checks": {
            "zero_delay_identity": zero_map_error < 1e-12,
            "zero_delay_spectral_identity": zero_rho_error < 1e-12,
            "direct_lift_agreement": direct_relative_error < 1e-12,
        },
        "limiting_point_refinements": refinements,
    }
    base.save_json(OUT / "validation.json", validation)

    comparison = [
        {
            "delay_ms": 0.0,
            "controller": "Current",
            "stable_count": 170,
            "feasible_count": 170,
            "non_degrading_count": 170,
            "minimum_rear_margin_percent": zero["current"]["minimum_rear_margin_81_percent"],
            "worst_front_LTR": zero["current"]["worst_front_LTR_21phase"],
            "max_constraint_utilization": zero["current"]["max_constraint_utilization_21phase"],
            "max_spectral_radius": zero["current"]["max_spectral_radius"],
        },
        {
            "delay_ms": 0.0,
            "controller": "5% reserve",
            "stable_count": 170,
            "feasible_count": 170,
            "non_degrading_count": 170,
            "minimum_rear_margin_percent": zero["reserve_5pct"]["minimum_rear_margin_81_percent"],
            "worst_front_LTR": zero["reserve_5pct"]["worst_front_LTR_21phase"],
            "max_constraint_utilization": zero["reserve_5pct"]["max_constraint_utilization_21phase"],
            "max_spectral_radius": zero["reserve_5pct"]["max_spectral_radius"],
        },
        {
            "delay_ms": 5.0,
            "controller": "Current",
            "stable_count": summary["stability"]["stable_count"],
            "feasible_count": summary["current"]["constraint_feasible_count"],
            "non_degrading_count": summary["current"]["non_degrading_count"],
            "minimum_rear_margin_percent": summary["current"]["minimum_rear_margin_percent"],
            "worst_front_LTR": summary["current"]["worst_front_LTR"],
            "max_constraint_utilization": summary["current"]["max_constraint_utilization"],
            "max_spectral_radius": summary["stability"]["maximum_spectral_radius"],
        },
        {
            "delay_ms": 5.0,
            "controller": "5% reserve",
            "stable_count": summary["stability"]["stable_count"],
            "feasible_count": summary["reserve_5pct"]["constraint_feasible_count"],
            "non_degrading_count": summary["reserve_5pct"]["non_degrading_count"],
            "minimum_rear_margin_percent": summary["reserve_5pct"]["minimum_rear_margin_percent"],
            "worst_front_LTR": summary["reserve_5pct"]["worst_front_LTR"],
            "max_constraint_utilization": summary["reserve_5pct"]["max_constraint_utilization"],
            "max_spectral_radius": summary["stability"]["maximum_spectral_radius"],
        },
    ]
    with (OUT / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    base.save_json(OUT / "comparison.json", comparison)

    table_lines = []
    for row in comparison:
        table_lines.append(
            f"{row['delay_ms']:.0f} & {row['controller']} & "
            f"{row['stable_count']}/170 & {row['feasible_count']}/170 & "
            f"{row['non_degrading_count']}/170 & "
            f"{row['minimum_rear_margin_percent']:.3f} & "
            f"{row['worst_front_LTR']:.3e} & "
            f"{row['max_constraint_utilization']:.3e} & "
            f"{row['max_spectral_radius']:.3f} \\\\"
        )
    (OUT / "tables" / "delay_0_5ms.tex").write_text("\n".join(table_lines) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7), constrained_layout=True)
    rho = np.sort([row["spectral_radius"] for row in rows])
    axes[0, 0].plot(np.arange(1, len(rho) + 1), rho, color="#0072B2")
    axes[0, 0].axhline(1.0, color="#D55E00", linestyle="--")
    axes[0, 0].set_xlabel("Sorted evaluated plant")
    axes[0, 0].set_ylabel("Spectral radius")
    axes[0, 0].set_title("(a) Stability at 5 ms")
    axes[0, 0].grid(alpha=0.25)

    x = np.arange(2)
    width = 0.34
    current_margin = [comparison[0]["minimum_rear_margin_percent"],
                      comparison[2]["minimum_rear_margin_percent"]]
    reserve_margin = [comparison[1]["minimum_rear_margin_percent"],
                      comparison[3]["minimum_rear_margin_percent"]]
    axes[0, 1].bar(x - width / 2, current_margin, width, label="Current",
                   color="#555555")
    axes[0, 1].bar(x + width / 2, reserve_margin, width, label="5% reserve",
                   color="#009E73")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.9)
    axes[0, 1].set_xticks(x, ["0 ms", "5 ms"])
    axes[0, 1].set_ylabel("Minimum rear margin (%)")
    axes[0, 1].set_title("(b) Rear non-degradation")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", alpha=0.25)

    front_current = [comparison[0]["worst_front_LTR"], comparison[2]["worst_front_LTR"]]
    front_reserve = [comparison[1]["worst_front_LTR"], comparison[3]["worst_front_LTR"]]
    axes[1, 0].bar(x - width / 2, front_current, width, color="#555555")
    axes[1, 0].bar(x + width / 2, front_reserve, width, color="#009E73")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(x, ["0 ms", "5 ms"])
    axes[1, 0].set_ylabel("Worst front LTR bound")
    axes[1, 0].set_title("(c) Four-second finite-horizon bound")
    axes[1, 0].grid(axis="y", alpha=0.25)

    util_current = [comparison[0]["max_constraint_utilization"],
                    comparison[2]["max_constraint_utilization"]]
    util_reserve = [comparison[1]["max_constraint_utilization"],
                    comparison[3]["max_constraint_utilization"]]
    axes[1, 1].bar(x - width / 2, util_current, width, color="#555555")
    axes[1, 1].bar(x + width / 2, util_reserve, width, color="#009E73")
    axes[1, 1].axhline(1.0, color="#D55E00", linestyle="--")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(x, ["0 ms", "5 ms"])
    axes[1, 1].set_ylabel("Maximum normalized utilization")
    axes[1, 1].set_title("(d) Constraint feasibility")
    axes[1, 1].grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "figures" / "delay_0_5ms_comparison.png", dpi=220)
    fig.savefig(OUT / "figures" / "delay_0_5ms_comparison.pdf")
    plt.close(fig)

    refined = refinements[0]["meshes"]["321"]
    scientific = (
        "# Five-millisecond latency audit\n\n"
        "A common 5 ms delay was applied to the full-state feedback and every "
        "steering-compensator tap while retaining the 20 ms controller period. "
        "The exact delayed lifting agrees with direct recurrence to a relative "
        f"error of {direct_relative_error:.2e}.\n\n"
        "The delayed autonomous loop is unstable at all 170 evaluated plants. "
        f"Its spectral radius ranges from {summary['stability']['minimum_spectral_radius']:.3f} "
        f"to {summary['stability']['maximum_spectral_radius']:.3f}. Consequently, "
        "neither controller satisfies the actuator/output constraints or rear "
        "non-degradation test at any evaluated plant. The minimum 81-phase rear "
        f"margins are {summary['current']['minimum_rear_margin_percent']:.2f}% "
        f"and {summary['reserve_5pct']['minimum_rear_margin_percent']:.2f}% for "
        "the current and reserve controllers, respectively. At the limiting "
        "point, the 321-phase values remain "
        f"{refined['current']['relative_margin_percent']:.2f}% and "
        f"{refined['reserve_5pct']['relative_margin_percent']:.2f}%.\n\n"
        "The very large four-second LTR and actuator bounds are mathematical "
        "consequences of the unstable linear delayed loop; they should not be "
        "interpreted as physically realizable nonlinear vehicle responses. The "
        "5% FIR reserve cannot restore stability because it does not alter the "
        "autonomous feedback dynamics. A delay-aware feedback redesign, predictor "
        "or observer is required before a nonzero-delay robustness claim can be made.\n"
    )
    (OUT / "scientific_summary.md").write_text(scientific)

    artifacts = [
        RESULTS, OUT / "validation.json", OUT / "comparison.csv",
        OUT / "comparison.json", OUT / "tables" / "delay_0_5ms.tex",
        OUT / "figures" / "delay_0_5ms_comparison.png",
        OUT / "figures" / "delay_0_5ms_comparison.pdf",
        OUT / "scientific_summary.md",
    ]
    manifest = {str(path.relative_to(HERE)): sha256(path) for path in artifacts}
    base.save_json(OUT / "artifact_manifest.json", manifest)
    print(json.dumps({
        "status": summary["status"],
        "validation": validation["checks"],
        "stable_count": summary["stability"]["stable_count"],
        "current_min_margin_percent": summary["current"]["minimum_rear_margin_percent"],
        "reserve_min_margin_percent": summary["reserve_5pct"]["minimum_rear_margin_percent"],
        "refined_321_current_percent": refined["current"]["relative_margin_percent"],
        "refined_321_reserve_percent": refined["reserve_5pct"]["relative_margin_percent"],
    }, indent=2))


if __name__ == "__main__":
    main()