"""Generate version-4 manuscript figures and LaTeX tables from saved results."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

COLORS = {
    "baseline": "#4D4D4D",
    "fixed_balanced": "#0072B2",
    "fixed_rear_protected": "#D55E00",
    "scheduled_balanced": "#009E73",
}
LABELS = {
    "baseline": "Fixed LQR1",
    "fixed_balanced": "Fixed balanced",
    "fixed_rear_protected": "Fixed rear-protected",
    "scheduled_balanced": "Speed scheduled",
}


def savefig(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def point_text(row):
    return (f"{row['speed_kmh']:.2f} & {row['mass_fraction']:.4f} & "
            f"{row['payload_height']:.4f}")


def main():
    result = json.loads((OUT / "envelope_results.json").read_text())
    verification = json.loads((OUT / "envelope_verification.json").read_text())
    design = json.loads((OUT / "fixed_rear_protected_design.json").read_text())
    rows = result["rows"]
    controllers = (
        "baseline", "fixed_balanced", "fixed_rear_protected",
        "scheduled_balanced",
    )

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.7,
        "figure.dpi": 140,
    })

    # Figure 1: operating-envelope evidence in four complementary panels.
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.45), constrained_layout=True)
    ax = axes[0, 0]
    structured = [r for r in rows if r["set"] == "structured"]
    speeds = sorted({r["speed_kmh"] for r in structured})
    for controller in controllers:
        values = [max(r[controller]["worst_LTR"] for r in structured
                      if r["speed_kmh"] == speed) for speed in speeds]
        ax.plot(speeds, values, marker="o", ms=3.7,
                color=COLORS[controller], label=LABELS[controller])
    ax.set_title("(a) Worst over structured payload cases")
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Worst-axle upper bound")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    sobol = [r for r in rows if r["set"] == "sobol_holdout"]
    for controller in controllers:
        values = np.sort([r[controller]["worst_LTR"] for r in sobol])
        probability = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, probability, where="post", color=COLORS[controller],
                label=LABELS[controller])
    ax.set_title("(b) Unseen-parameter empirical CDF")
    ax.set_xlabel("Worst-axle upper bound")
    ax.set_ylabel("Fraction of 128 Sobol points")
    ax.grid(alpha=.25)

    ax = axes[1, 0]
    fine = verification["rear_rows"]
    sc = ax.scatter([r["speed_kmh"] for r in fine],
                    [1e4 * r["certified_margin"] for r in fine],
                    c=[r["mass_fraction"] for r in fine], cmap="viridis",
                    s=16, alpha=.78, edgecolors="none")
    ax.axhline(0.0, color="black", lw=.9)
    ax.set_title("(c) One-sided rear-axle certificate")
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel(r"Certified margin ($\times 10^{-4}$)")
    cb = fig.colorbar(sc, ax=ax, pad=.015)
    cb.set_label("Added mass fraction")
    ax.grid(alpha=.2)

    ax = axes[1, 1]
    width = 1.6
    offsets = np.linspace(-2.4, 2.4, 3)
    for offset, controller in zip(offsets,
                                  controllers[1:]):
        values = [max(r[controller]["constraint_utilization"]
                      for r in structured if r["speed_kmh"] == speed)
                  for speed in speeds]
        ax.bar(np.asarray(speeds) + offset, values, width=width,
               color=COLORS[controller], alpha=.86)
    ax.axhline(1.0, color="black", ls="--", lw=1.0)
    ax.text(50.0, 1.0015, "limit", ha="center", va="bottom", fontsize=7.5)
    ax.set_title("(d) Largest non-LTR utilization")
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Normalized utilization")
    ax.set_ylim(.78, 1.015)
    ax.grid(axis="y", alpha=.25)
    savefig(fig, "operating_envelope_summary")

    # Figure 2: high-payload maps at structured grid points.
    loaded = [r for r in structured
              if abs(r["mass_fraction"] - 0.10) < 1e-12
              and abs(r["payload_height"] - 1.80) < 1e-12]
    loaded.sort(key=lambda r: r["speed_kmh"])
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25), constrained_layout=True)
    for ax, controller, title in zip(
        axes[:2], ("fixed_rear_protected", "scheduled_balanced"),
        ("Fixed rear-protected", "Speed scheduled"),
    ):
        ax.plot([r["speed_kmh"] for r in loaded],
                [r[controller]["front_LTR"] for r in loaded],
                marker="o", color="#CC3311", label="Front")
        ax.plot([r["speed_kmh"] for r in loaded],
                [r[controller]["rear_LTR"] for r in loaded],
                marker="s", color="#0077BB", label="Rear")
        ax.set_title(title)
        ax.set_xlabel("Speed (km/h)")
        ax.set_ylabel("Axle upper bound")
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    ax = axes[2]
    all_speeds = np.asarray([r["speed_kmh"] for r in sobol])
    improvement = np.asarray([r["schedule_improvement_percent"] for r in sobol])
    scatter = ax.scatter(all_speeds, improvement,
                         c=[r["mass_fraction"] for r in sobol],
                         cmap="plasma", s=18, alpha=.8, edgecolors="none")
    ax.axhline(0.0, color="black", lw=.9)
    ax.set_title("Scheduled vs fixed balanced")
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Improvement (%)")
    ax.grid(alpha=.2)
    cb = fig.colorbar(scatter, ax=ax, pad=.015)
    cb.set_label("Added mass fraction")
    savefig(fig, "envelope_tradeoffs")

    # Figure 3: direct graphical form of the one-sided proof.
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), constrained_layout=True)
    ax = axes[0]
    baseline_lower = np.asarray(
        [r["baseline"]["grid_feasible_lower"] for r in fine]
    )
    proposed_upper = np.asarray(
        [r["proposed"]["continuous_upper"] for r in fine]
    )
    ax.scatter(baseline_lower, proposed_upper, color="#336699",
               s=17, alpha=.72, edgecolors="none")
    lo = min(baseline_lower.min(), proposed_upper.min())
    hi = max(baseline_lower.max(), proposed_upper.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="Equality")
    ax.set_xlabel("LQR1 feasible sampled lower bound")
    ax.set_ylabel("Proposed continuous upper bound")
    ax.set_title("(a) All 170 rear-axle comparisons")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    ax = axes[1]
    margin = 100.0 * np.asarray(
        [r["certified_margin"] /
         r["baseline"]["grid_feasible_lower"] for r in fine]
    )
    ax.hist(margin, bins=18, color="#D55E00", alpha=.82, edgecolor="white")
    ax.axvline(margin.min(), color="black", ls="--", lw=1.1,
               label=f"Minimum = {margin.min():.3f}%")
    ax.set_xlabel("Rear-certificate margin relative to LQR1 (%)")
    ax.set_ylabel("Parameter points")
    ax.set_title("(b) Distribution of certified margins")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    savefig(fig, "rear_certificate")

    baseline_worst = result["baseline_worst_LTR"]
    baseline_rho = max(r["baseline"]["spectral_radius"] for r in rows)
    baseline_util = max(r["baseline"]["constraint_utilization"] for r in rows)
    summary_lines = []
    for controller, rear in (
        ("baseline", "Reference"),
        ("fixed_balanced", "No"),
        ("fixed_rear_protected", "Yes"),
        ("scheduled_balanced", "No"),
    ):
        if controller == "baseline":
            worst, util, rho = baseline_worst, baseline_util, baseline_rho
        else:
            item = result["controllers"][controller]
            worst = item["worst_LTR"]
            util = item["max_constraint_utilization"]
            rho = item["max_spectral_radius"]
        reduction = 100.0 * (1.0 - worst / baseline_worst)
        summary_lines.append(
            f"{LABELS[controller]} & {worst:.6f} & {reduction:.2f} & "
            f"{util:.6f} & {rho:.6f} & {rear} \\\\"
        )
    (TAB / "envelope_summary.tex").write_text("\n".join(summary_lines) + "\n")

    point_lines = []
    baseline_point = result["baseline_worst_point"]
    point_lines.append(
        f"Fixed LQR1 & {point_text(baseline_point)} & "
        f"{baseline_worst:.6f} \\\\"
    )
    for controller in controllers[1:]:
        info = result["controllers"][controller]
        point_lines.append(
            f"{LABELS[controller]} & {point_text(info['worst_LTR_point'])} & "
            f"{info['worst_LTR']:.6f} \\\\"
        )
    (TAB / "worst_points.tex").write_text("\n".join(point_lines) + "\n")

    sobol = [r for r in rows if r["set"] == "sobol_holdout"]
    sensitivity_lines = []
    for controller in controllers:
        y = [r[controller]["worst_LTR"] for r in sobol]
        correlations = []
        for variable in ("speed_kmh", "mass_fraction", "payload_height"):
            rho, _ = spearmanr([r[variable] for r in sobol], y)
            correlations.append(rho)
        sensitivity_lines.append(
            f"{LABELS[controller]} & {correlations[0]:.3f} & "
            f"{correlations[1]:.3f} & {correlations[2]:.3f} \\\\"
        )
    (TAB / "sensitivity.tex").write_text("\n".join(sensitivity_lines) + "\n")

    worst_set_lines = []
    for set_name, printed in (("structured", "Structured grid"),
                              ("sobol_holdout", "Sobol holdout")):
        subset = [r for r in rows if r["set"] == set_name]
        values = [max(r[c]["worst_LTR"] for r in subset) for c in controllers]
        worst_set_lines.append(
            f"{printed} & {values[0]:.6f} & {values[1]:.6f} & "
            f"{values[2]:.6f} & {values[3]:.6f} \\\\"
        )
    (TAB / "structured_holdout.tex").write_text("\n".join(worst_set_lines) + "\n")

    coefficients = np.asarray(design["coefficients"]).reshape(2, 6)
    delays = (0, 2, 5, 10, 20, 40)
    coefficient_lines = []
    for index, delay in enumerate(delays):
        coefficient_lines.append(
            f"{delay} & {0.02*delay:.2f} & {coefficients[0,index]:.9f} & "
            f"{coefficients[1,index]:.9f} \\\\"
        )
    (TAB / "v4_coefficients.tex").write_text("\n".join(coefficient_lines) + "\n")

    final_summary = {
        "status": "PASS" if (
            verification["status"] == "PASS"
            and verification["rear_all_certified"]
            and all(result["checks"][name] for name in (
                "fixed_balanced_constraints", "fixed_protected_constraints",
                "scheduled_constraints", "fixed_stability", "scheduled_stability",
            ))
        ) else "REVIEW",
        "date": "2026-09-19",
        "evaluated_parameter_points": result["count"],
        "broad_screening_phases": result["protocol"]["phases"],
        "fine_rear_audit_phases": verification["phases"],
        "rear_all_certified": verification["rear_all_certified"],
        "minimum_rear_certificate_margin": verification["minimum_certified_margin"],
        "constraint_and_stability_checks": {
            name: result["checks"][name] for name in (
                "fixed_balanced_constraints", "fixed_protected_constraints",
                "scheduled_constraints", "fixed_stability", "scheduled_stability",
            )
        },
        "note": (
            "The coarse independent-upper-bound rear ratio is a screening "
            "quantity. Final rear non-degradation is decided by the 81-phase "
            "one-sided upper-versus-feasible-lower certificate."
        ),
    }
    (OUT / "stage4_final_summary.json").write_text(
        json.dumps(final_summary, indent=2, sort_keys=True) + "\n"
    )

    manifest = {}
    for path in sorted(list(FIG.glob("*")) + list(TAB.glob("*"))):
        manifest[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "artifact_manifest_v4.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()