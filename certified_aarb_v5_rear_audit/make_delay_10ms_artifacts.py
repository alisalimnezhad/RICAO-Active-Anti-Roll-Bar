"""Refine and package the 10 ms latency audit with 0/5/10 ms comparison."""
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


OUT = HERE / "outputs" / "delay_sensitivity" / "delay_10ms"
ZERO = HERE / "outputs" / "delay_sensitivity" / "delay_0ms" / "results.json"
FIVE = HERE / "outputs" / "delay_sensitivity" / "delay_5ms" / "results.json"
TEN = OUT / "results.json"
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


def rows_for_delay(delay, summary):
    if delay == 0.0:
        return [
            {
                "delay_ms": 0.0, "controller": "Current",
                "stable_count": 170, "feasible_count": 170,
                "non_degrading_count": 170,
                "minimum_rear_margin_percent": summary["current"]["minimum_rear_margin_81_percent"],
                "worst_front_LTR": summary["current"]["worst_front_LTR_21phase"],
                "max_constraint_utilization": summary["current"]["max_constraint_utilization_21phase"],
                "max_spectral_radius": summary["current"]["max_spectral_radius"],
            },
            {
                "delay_ms": 0.0, "controller": "5% reserve",
                "stable_count": 170, "feasible_count": 170,
                "non_degrading_count": 170,
                "minimum_rear_margin_percent": summary["reserve_5pct"]["minimum_rear_margin_81_percent"],
                "worst_front_LTR": summary["reserve_5pct"]["worst_front_LTR_21phase"],
                "max_constraint_utilization": summary["reserve_5pct"]["max_constraint_utilization_21phase"],
                "max_spectral_radius": summary["reserve_5pct"]["max_spectral_radius"],
            },
        ]
    return [
        {
            "delay_ms": delay, "controller": "Current",
            "stable_count": summary["stability"]["stable_count"],
            "feasible_count": summary["current"]["constraint_feasible_count"],
            "non_degrading_count": summary["current"]["non_degrading_count"],
            "minimum_rear_margin_percent": summary["current"]["minimum_rear_margin_percent"],
            "worst_front_LTR": summary["current"]["worst_front_LTR"],
            "max_constraint_utilization": summary["current"]["max_constraint_utilization"],
            "max_spectral_radius": summary["stability"]["maximum_spectral_radius"],
        },
        {
            "delay_ms": delay, "controller": "5% reserve",
            "stable_count": summary["stability"]["stable_count"],
            "feasible_count": summary["reserve_5pct"]["constraint_feasible_count"],
            "non_degrading_count": summary["reserve_5pct"]["non_degrading_count"],
            "minimum_rear_margin_percent": summary["reserve_5pct"]["minimum_rear_margin_percent"],
            "worst_front_LTR": summary["reserve_5pct"]["worst_front_LTR"],
            "max_constraint_utilization": summary["reserve_5pct"]["max_constraint_utilization"],
            "max_spectral_radius": summary["stability"]["maximum_spectral_radius"],
        },
    ]


def main():
    zero = json.loads(ZERO.read_text())["summary"]
    five = json.loads(FIVE.read_text())["summary"]
    ten_record = json.loads(TEN.read_text())
    ten = ten_record["summary"]
    ten_rows = ten_record["rows"]
    current_f = np.loadtxt(CURRENT_COEFFICIENTS, delimiter=",").reshape(-1)
    reserve_f = np.loadtxt(RESERVE_COEFFICIENTS, delimiter=",").reshape(-1)
    controllers = {
        "baseline": np.zeros(12),
        "current": current_f,
        "reserve_5pct": reserve_f,
    }

    nominal = ParametricStudy(speed_kmh=70.0, phases=9)
    fixed_gain = nominal.K.copy()
    lookup = {point_key(row): row for row in ten_rows}
    limit_keys = {
        point_key(ten["current"]["limiting_rear_point"]),
        point_key(ten["reserve_5pct"]["limiting_rear_point"]),
    }
    refinements = []
    for key in sorted(limit_keys):
        point = lookup[key]
        study = DelayedParametricStudy(
            speed_kmh=point["speed_kmh"],
            overrides=distributed_payload_overrides(
                point["mass_fraction"], point["payload_height"]
            ),
            feedback_gain=fixed_gain, delay_s=0.010, phases=9,
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

    phases = np.linspace(0.0, 0.02, 11)
    rng = np.random.default_rng(20260922)
    increments = rng.uniform(-0.05, 0.05, 200)
    steering = np.clip(np.cumsum(increments), -1.0, 1.0)
    test_f = rng.uniform(-3.0, 3.0, 12)
    delayed = DelayedParametricStudy(speed_kmh=70.0, delay_s=0.010, phases=9)
    lifted = np.einsum(
        "pin,n->pi", delayed.combine_maps(delayed.maps(phases), test_f), steering
    )
    direct = delayed.direct_final_interval(test_f, steering, phases)
    direct_error = float(
        np.max(np.abs(lifted - direct)) /
        max(1.0, float(np.max(np.abs(lifted))), float(np.max(np.abs(direct))))
    )
    validation = {
        "date": "2026-09-22",
        "ten_ms_direct_lift_max_relative_error": direct_error,
        "nominal_ten_ms_spectral_radius": delayed.rho_closed,
        "direct_lift_agreement": direct_error < 1e-12,
        "limiting_point_refinements": refinements,
    }
    base.save_json(OUT / "validation.json", validation)

    comparison = (
        rows_for_delay(0.0, zero)
        + rows_for_delay(5.0, five)
        + rows_for_delay(10.0, ten)
    )
    with (OUT / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    base.save_json(OUT / "comparison.json", comparison)

    table = []
    for row in comparison:
        table.append(
            f"{row['delay_ms']:.0f} & {row['controller']} & "
            f"{row['stable_count']}/170 & {row['feasible_count']}/170 & "
            f"{row['non_degrading_count']}/170 & "
            f"{row['minimum_rear_margin_percent']:.3f} & "
            f"{row['worst_front_LTR']:.3e} & "
            f"{row['max_constraint_utilization']:.3e} & "
            f"{row['max_spectral_radius']:.3f} \\\\"
        )
    (OUT / "tables" / "delay_0_5_10ms.tex").write_text("\n".join(table) + "\n")

    delay_values = [0.0, 5.0, 10.0]
    current = [comparison[0], comparison[2], comparison[4]]
    reserve = [comparison[1], comparison[3], comparison[5]]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7), constrained_layout=True)
    axes[0, 0].plot(delay_values, [r["max_spectral_radius"] for r in current],
                    marker="o", color="#0072B2")
    axes[0, 0].axhline(1.0, color="#D55E00", linestyle="--")
    axes[0, 0].set_ylabel("Maximum spectral radius")
    axes[0, 0].set_title("(a) Delayed-loop stability")
    axes[0, 0].set_xticks(delay_values)
    axes[0, 0].grid(alpha=0.25)

    x = np.arange(3); width = 0.34
    axes[0, 1].bar(x-width/2, [r["minimum_rear_margin_percent"] for r in current],
                   width, label="Current", color="#555555")
    axes[0, 1].bar(x+width/2, [r["minimum_rear_margin_percent"] for r in reserve],
                   width, label="5% reserve", color="#009E73")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.9)
    axes[0, 1].set_xticks(x, ["0", "5", "10"])
    axes[0, 1].set_ylabel("Minimum rear margin (%)")
    axes[0, 1].set_title("(b) Rear non-degradation")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(x-width/2, [r["worst_front_LTR"] for r in current],
                   width, color="#555555")
    axes[1, 0].bar(x+width/2, [r["worst_front_LTR"] for r in reserve],
                   width, color="#009E73")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(x, ["0", "5", "10"])
    axes[1, 0].set_xlabel("Measurement delay (ms)")
    axes[1, 0].set_ylabel("Worst front LTR bound")
    axes[1, 0].set_title("(c) Four-second bound")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].bar(x-width/2, [r["max_constraint_utilization"] for r in current],
                   width, color="#555555")
    axes[1, 1].bar(x+width/2, [r["max_constraint_utilization"] for r in reserve],
                   width, color="#009E73")
    axes[1, 1].axhline(1.0, color="#D55E00", linestyle="--")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(x, ["0", "5", "10"])
    axes[1, 1].set_xlabel("Measurement delay (ms)")
    axes[1, 1].set_ylabel("Maximum normalized utilization")
    axes[1, 1].set_title("(d) Constraint feasibility")
    axes[1, 1].grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "figures" / "delay_0_5_10ms_comparison.png", dpi=220)
    fig.savefig(OUT / "figures" / "delay_0_5_10ms_comparison.pdf")
    plt.close(fig)

    refined = refinements[0]["meshes"]["321"]
    scientific = (
        "# Ten-millisecond latency audit\n\n"
        "The common state/steering measurement delay was increased to 10 ms "
        "while the controller period remained 20 ms. The exact lifted maps "
        f"agree with direct recurrence to {direct_error:.2e} in relative terms.\n\n"
        "All 170 evaluated plants are unstable. The delayed-loop spectral "
        f"radius ranges from {ten['stability']['minimum_spectral_radius']:.3f} "
        f"to {ten['stability']['maximum_spectral_radius']:.3f}, compared with a "
        f"maximum of {five['stability']['maximum_spectral_radius']:.3f} at 5 ms. "
        "No plant satisfies the declared constraints or rear non-degradation "
        "condition for either controller. The 81-phase minimum rear margins "
        f"are {ten['current']['minimum_rear_margin_percent']:.2f}% and "
        f"{ten['reserve_5pct']['minimum_rear_margin_percent']:.2f}%; the "
        "321-phase limiting-point values are "
        f"{refined['current']['relative_margin_percent']:.2f}% and "
        f"{refined['reserve_5pct']['relative_margin_percent']:.2f}%.\n\n"
        "Although the relative rear margin is less negative than at 5 ms, the "
        "absolute four-second bounds are vastly larger because the delayed loop "
        "is more unstable. Therefore the relative comparison must not be read as "
        "an improvement. The FIR reserve does not affect autonomous stability; "
        "delay-aware feedback redesign is required.\n"
    )
    (OUT / "scientific_summary.md").write_text(scientific)

    artifacts = [
        TEN, OUT / "validation.json", OUT / "comparison.csv",
        OUT / "comparison.json", OUT / "tables" / "delay_0_5_10ms.tex",
        OUT / "figures" / "delay_0_5_10ms_comparison.png",
        OUT / "figures" / "delay_0_5_10ms_comparison.pdf",
        OUT / "scientific_summary.md",
    ]
    base.save_json(OUT / "artifact_manifest.json", {
        str(path.relative_to(HERE)): sha256(path) for path in artifacts
    })
    print(json.dumps({
        "status": ten["status"],
        "direct_lift_agreement": validation["direct_lift_agreement"],
        "stable_count": ten["stability"]["stable_count"],
        "current_min_margin_percent": ten["current"]["minimum_rear_margin_percent"],
        "reserve_min_margin_percent": ten["reserve_5pct"]["minimum_rear_margin_percent"],
        "refined_321_current_percent": refined["current"]["relative_margin_percent"],
        "refined_321_reserve_percent": refined["reserve_5pct"]["relative_margin_percent"],
    }, indent=2))


if __name__ == "__main__":
    main()