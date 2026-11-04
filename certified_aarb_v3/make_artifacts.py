"""Create version-3 vector figures and LaTeX tables from saved results."""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

stage = json.loads((OUT / "stage3_results.json").read_text())
verify = json.loads((OUT / "stage3_verification.json").read_text())

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COL = {
    "base": "#444444", "balanced": "#00798c", "protected": "#b23a48",
    "scheduled": "#5b3f95", "v2": "#d98e04", "limit": "#777777",
}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def c(record, key):
    return float(record["continuous"][key])


# Figure 1: nominal Pareto frontier and fair-current points.
pareto = stage["nominal_pareto"]
px = np.array([r["worst_continuous_rear_LTR"] for r in pareto])
py = np.array([r["worst_continuous_front_LTR"] for r in pareto])
baseline = verify["nominal_baseline"]["continuous"]
fair_b = verify["nominal_designs"]["fair_current_balanced"]["continuous"]
fair_p = verify["nominal_designs"]["fair_current_rear_protected"]["continuous"]
full_p = verify["nominal_designs"]["pareto_0"]["continuous"]

fig, ax = plt.subplots(figsize=(5.35, 3.45))
ax.plot(px, py, "o-", color=COL["protected"], lw=1.6, ms=4.5,
        label="20 mA certified Pareto designs")
ax.scatter([baseline["rear_LTR"]], [baseline["front_LTR"]], marker="s", s=45,
           color=COL["base"], zorder=4, label="Sampled LQR1")
ax.scatter([fair_b["rear_LTR"]], [fair_b["front_LTR"]], marker="^", s=50,
           color=COL["balanced"], zorder=4, label="LQR1-current caps: balanced")
ax.scatter([fair_p["rear_LTR"]], [fair_p["front_LTR"]], marker="D", s=42,
           color=COL["scheduled"], zorder=4, label="LQR1-current caps: rear protected")
ax.scatter([full_p["rear_LTR"]], [full_p["front_LTR"]], marker="*", s=90,
           color=COL["v2"], edgecolor="black", linewidth=.35, zorder=5,
           label="20 mA: rear protected")
ax.axvline(baseline["rear_LTR"], color=COL["base"], lw=.8, ls="--")
ax.set_xlabel(r"Certified rear-axle bound $U_R$")
ax.set_ylabel(r"Certified front-axle bound $U_F$")
ax.set_title("Nominal axle trade-off and current-matched designs")
ax.grid(True, alpha=.20)
ax.legend(frameon=False, loc="upper right")
save(fig, "nominal_pareto")


# Figure 2: output/current comparison for the key nominal controllers.
labels = ["LQR1", "Matched\nbalanced", "Matched\nrear-safe",
          "20 mA\nrear-safe"]
records = [baseline, fair_b, fair_p, full_p]
front = [r["front_LTR"] for r in records]
rear = [r["rear_LTR"] for r in records]
ifront = [20 * r["front_current"] for r in records]
irear = [20 * r["rear_current"] for r in records]
x = np.arange(len(labels)); width = .34
fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.15))
axes[0].bar(x-width/2, front, width, color="#2a6fbb", label="Front")
axes[0].bar(x+width/2, rear, width, color="#d1495b", label="Rear")
axes[0].set_xticks(x, labels)
axes[0].tick_params(axis="x", labelsize=7.5)
axes[0].set_ylabel("Certified LTR bound")
axes[0].set_title("Axle-wise transient bounds")
axes[0].legend(frameon=False)
axes[0].grid(axis="y", alpha=.2)
axes[1].bar(x-width/2, ifront, width, color="#2a6fbb", label="Front")
axes[1].bar(x+width/2, irear, width, color="#d1495b", label="Rear")
axes[1].axhline(20, color=COL["limit"], ls="--", lw=1, label="Rated limit")
axes[1].set_xticks(x, labels)
axes[1].tick_params(axis="x", labelsize=7.5)
axes[1].set_ylabel("Certified peak current (mA)")
axes[1].set_title("Actual peak-current authority")
axes[1].grid(axis="y", alpha=.2)
axes[1].legend(frameon=False)
save(fig, "nominal_fairness")


# Figure 3: model-bank worst-axle comparison by case.
bank = verify["bank"]
sched = {r["label"]: r for r in verify["scheduled_bank"]}
short_load = {"nominal": "N", "uniform-mass-plus-10pct": "U10",
              "central-high-payload-plus-10pct": "H10"}
case_labels, base_values, fixed_values, protected_values, sched_values = [], [], [], [], []
for row in bank:
    speed, load = row["label"].split(" km/h; ")
    case_labels.append(f"{speed}-{short_load[load]}")
    for key, bucket in (("baseline", base_values), ("fixed_balanced", fixed_values),
                        ("fixed_rear_protected", protected_values)):
        cr = row[key]["continuous"]
        bucket.append(max(cr["front_LTR"], cr["rear_LTR"]))
    cr = sched[row["label"]]["scheduled"]["continuous"]
    sched_values.append(max(cr["front_LTR"], cr["rear_LTR"]))

x = np.arange(len(case_labels))
fig, ax = plt.subplots(figsize=(7.1, 3.5))
ax.plot(x, base_values, "o-", color=COL["base"], label="Fixed LQR1")
ax.plot(x, fixed_values, "o-", color=COL["balanced"], label="Fixed robust, balanced")
ax.plot(x, protected_values, "o-", color=COL["protected"], label="Fixed robust, rear protected")
ax.plot(x, sched_values, "o-", color=COL["scheduled"], label="Speed scheduled, balanced")
ax.set_xticks(x, case_labels)
ax.set_ylabel("Certified worst-axle bound")
ax.set_xlabel("Speed-load case (N: nominal, U10: uniform +10%, H10: high-CG +10%)")
ax.set_title("Finite nine-model bank: controller comparison")
ax.grid(True, alpha=.2)
ax.legend(frameon=False, ncol=2)
save(fig, "model_bank_comparison")


# Figure 4: value of scheduling by speed.
speeds = [50.0, 70.0, 75.0]
fixed_speed, sched_speed = [], []
for speed in speeds:
    prefix = f"{speed:g} km/h;"
    fb = [row for row in bank if row["label"].startswith(prefix)]
    sb = [row for row in verify["scheduled_bank"] if row["label"].startswith(prefix)]
    fixed_speed.append(max(max(c(row["fixed_balanced"], "front_LTR"),
                               c(row["fixed_balanced"], "rear_LTR")) for row in fb))
    sched_speed.append(max(max(c(row["scheduled"], "front_LTR"),
                               c(row["scheduled"], "rear_LTR")) for row in sb))
fig, ax = plt.subplots(figsize=(5.25, 3.25))
x = np.arange(3)
ax.bar(x-.18, fixed_speed, .36, color=COL["balanced"], label="One fixed controller")
ax.bar(x+.18, sched_speed, .36, color=COL["scheduled"], label="Speed-scheduled bank")
for i, (f, s) in enumerate(zip(fixed_speed, sched_speed)):
    ax.text(i, max(f, s)+.0012, f"{100*(1-s/f):.1f}%", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x, [f"{v:g}" for v in speeds])
ax.set_xlabel("Speed (km/h)")
ax.set_ylabel("Worst bound across three load cases")
ax.set_title("Certified value of speed scheduling")
ax.grid(axis="y", alpha=.2)
ax.legend(frameon=False, loc="lower left")
save(fig, "scheduling_value")


# Figure 5: analytical-domain screen.
domain = stage["domain_screen"]
fig, ax = plt.subplots(figsize=(5.45, 3.25))
load_order = ["nominal", "uniform-mass-plus-10pct", "central-high-payload-plus-10pct"]
styles = ["o-", "s-", "^-" ]; colors = ["#2a6fbb", "#d1495b", "#5b3f95"]
for load, style, color in zip(load_order, styles, colors):
    rows = [r for r in domain if r["label"].endswith(load)]
    rows.sort(key=lambda r: r["speed_kmh"])
    ax.plot([r["speed_kmh"] for r in rows],
            [r["maximum_non_LTR_utilization"] for r in rows], style,
            color=color, label=short_load[load])
ax.axhline(1.0, color=COL["limit"], ls="--", lw=1, label="Declared domain limit")
ax.set_xlabel("Speed (km/h)")
ax.set_ylabel("Largest normalized non-LTR output")
ax.set_title("Predeclared tyre/actuator-domain screen for fixed LQR1")
ax.grid(True, alpha=.2)
ax.legend(frameon=False, ncol=2)
save(fig, "domain_screen")


# Tables.
base_front = baseline["front_LTR"]
nominal_rows = [
    ("Sampled LQR1", baseline),
    ("Equal-current balanced", fair_b),
    ("Equal-current rear-protected", fair_p),
    ("20 mA rear-protected", full_p),
]
with (TAB / "nominal_fairness.tex").open("w") as f:
    for name, row in nominal_rows:
        worst = max(row["front_LTR"], row["rear_LTR"])
        reduction = 100 * (1 - row["front_LTR"] / base_front)
        f.write(f"{name} & {row['front_LTR']:.6f} & {row['rear_LTR']:.6f} & "
                f"{20*row['front_current']:.4f} & {20*row['rear_current']:.4f} & "
                f"{worst:.6f} & {reduction:.2f} \\\\\n")

with (TAB / "pareto.tex").open("w") as f:
    for row in pareto:
        f.write(f"{row['rear_target']:.6f} & {row['worst_continuous_front_LTR']:.6f} & "
                f"{row['worst_continuous_rear_LTR']:.6f} & "
                f"{row['worst_front_current_mA']:.4f} & {row['worst_rear_current_mA']:.4f} \\\\\n")

bank_base = max(base_values)
bank_fixed = max(fixed_values)
bank_protected = max(protected_values)
bank_scheduled = max(sched_values)
with (TAB / "bank_summary.tex").open("w") as f:
    rows = [
        ("Fixed sampled LQR1", bank_base, "--", "Yes"),
        ("Fixed balanced robust", bank_fixed, f"{100*(1-bank_fixed/bank_base):.2f}", "No"),
        ("Fixed rear-protected robust", bank_protected, f"{100*(1-bank_protected/bank_base):.2f}", "Yes (all 9)"),
        ("Speed-scheduled balanced", bank_scheduled, f"{100*(1-bank_scheduled/bank_base):.2f}", "No"),
    ]
    for name, value, reduction, rear_ok in rows:
        f.write(f"{name} & {value:.6f} & {reduction} & {rear_ok} \\\\\n")

with (TAB / "scheduling.tex").open("w") as f:
    for speed, fv, sv in zip(speeds, fixed_speed, sched_speed):
        f.write(f"{speed:g} & {fv:.6f} & {sv:.6f} & {100*(1-sv/fv):.2f} \\\\\n")
    f.write(f"All speeds & {bank_fixed:.6f} & {bank_scheduled:.6f} & "
            f"{100*(1-bank_scheduled/bank_fixed):.2f} \\\\\n")

with (TAB / "domain.tex").open("w") as f:
    for speed in (75.0, 80.0, 90.0):
        rows = [r for r in domain if r["speed_kmh"] == speed]
        worst = max(r["maximum_non_LTR_utilization"] for r in rows)
        limiting = max(rows, key=lambda r: r["maximum_non_LTR_utilization"])["label"].split("; ")[1]
        f.write(f"{speed:g} & {worst:.6f} & {limiting} & "
                f"{'Admitted' if worst <= 1 else 'Excluded'} \\\\\n")

fixed_coeff = np.loadtxt(OUT / "bank_fixed_balanced_coefficients.csv", delimiter=",")
protected_coeff = np.loadtxt(OUT / "bank_fixed_rear_protected_coefficients.csv", delimiter=",")
delays = [0.00, 0.04, 0.10, 0.20, 0.40, 0.80]
with (TAB / "v3_coefficients.tex").open("w") as f:
    for j, delay in enumerate(delays):
        f.write(f"{delay:.2f} & {fixed_coeff[0,j]:.8f} & {fixed_coeff[1,j]:.8f} & "
                f"{protected_coeff[0,j]:.8f} & {protected_coeff[1,j]:.8f} \\\\\n")

manifest = {
    "figures": sorted(path.name for path in FIG.glob("*.pdf")),
    "tables": sorted(path.name for path in TAB.glob("*.tex")),
    "source_results": ["stage3_results.json", "stage3_verification.json"],
}
(OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))