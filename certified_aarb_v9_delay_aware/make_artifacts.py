"""Create publication figures and tables from version-9 JSON artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
PUB_FIG = ROOT / "paper" / "figures"
PUB_TAB = ROOT / "paper" / "tables"
PUB_FIG.mkdir(parents=True, exist_ok=True)
PUB_TAB.mkdir(parents=True, exist_ok=True)


def write(path, text):
    path.write_text(text)


def main():
    gain = json.loads((OUT / "gain_scan.json").read_text())
    linear = json.loads((OUT / "results.json").read_text())
    nonlinear = json.loads((OUT / "nonlinear" / "results.json").read_text())
    design = json.loads((OUT / "design.json").read_text())

    scales = np.array([row["scale"] for row in gain["rows"]])
    radii = np.array([row["maximum_spectral_radius"] for row in gain["rows"]])
    selected = float(gain["selected_scale"])
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.5))
    axes[0].plot(scales, radii, marker="o", ms=2.8, lw=1.3, color="#176B87")
    axes[0].axhline(1.0, color="#A33", ls="--", lw=1.0, label="Schur boundary")
    axes[0].axvline(selected, color="#D97706", ls=":", lw=1.3,
                    label=f"selected $\\alpha={selected:.3f}$")
    axes[0].set(xlabel="LQR1 gain scale $\\alpha$",
                ylabel="maximum spectral radius", title="(a) 170-plant gain search")
    axes[0].legend(frameon=False, fontsize=8)
    labels = ["zero-delay\n5% reserve", "5 ms\ndelay-aware"]
    zero = 0.0590920144
    vals = [zero, linear["summary"]["worst_LTR"]]
    axes[1].bar(labels, vals, color=["#8FAADC", "#2A9D8F"], width=0.58)
    axes[1].set(ylabel="evaluated worst LTR", title="(b) Pointwise bank maxima")
    axes[1].set_ylim(0, max(vals) * 1.28)
    for i, value in enumerate(vals):
        axes[1].text(i, value + 0.0012, f"{value:.5f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "delay_aware_design.pdf", bbox_inches="tight")
    fig.savefig(FIG / "delay_aware_design.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    cases = nonlinear["cases"]
    names = [
        "front witness", "rear witness", "tyre-slip witness",
        "payload witness", "fishhook",
    ]
    linear_front = [row["linear"]["peak_front_LTR"] for row in cases]
    nonlinear_front = [row["nonlinear"]["peak_front_LTR"] for row in cases]
    linear_rear = [row["linear"]["peak_rear_LTR"] for row in cases]
    nonlinear_rear = [row["nonlinear"]["peak_rear_LTR"] for row in cases]
    x = np.arange(len(cases))
    fig, axes = plt.subplots(2, 1, figsize=(9.3, 5.3), sharex=True)
    width = 0.36
    axes[0].bar(x-width/2, linear_front, width, label="linear", color="#8FAADC")
    axes[0].bar(x+width/2, nonlinear_front, width, label="nonlinear", color="#E76F51")
    axes[0].set_ylabel("peak front LTR")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].bar(x-width/2, linear_rear, width, label="linear", color="#8FAADC")
    axes[1].bar(x+width/2, nonlinear_rear, width, label="nonlinear", color="#E76F51")
    axes[1].set_ylabel("peak rear LTR")
    axes[1].set_xticks(x, names, rotation=18, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "nonlinear_critical_cases.pdf", bbox_inches="tight")
    fig.savefig(FIG / "nonlinear_critical_cases.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    summary = linear["summary"]
    delay_table = r"""\begin{tabular}{lrr}\toprule
Quantity & Fixed zero-delay gain at 5 ms & Delay-aware redesign\\\midrule
Stable plants (of 170) & 0 & %d\\
Maximum spectral radius & 1.21647 & %.5f\\
Worst evaluated LTR & -- & %.5f\\
Maximum constraint utilization & -- & %.3f\\
Rear non-degrading plants & -- & %d\\
\bottomrule\end{tabular}
""" % (
        summary["stable_count"], summary["maximum_spectral_radius"],
        summary["worst_LTR"], summary["max_constraint_utilization"],
        summary["rear_non_degrading_count"],
    )
    write(TAB / "delay_aware_summary.tex", delay_table)

    lines = [r"\begin{tabular}{lrrrr}\toprule",
             r"Case & Linear $R_F$ & Nonlinear $R_F$ & Linear $R_R$ & Nonlinear $R_R$\\\midrule"]
    latex_names = ["Front witness", "Rear witness", "Tyre-slip witness",
                   "Payload witness", "Fishhook"]
    for label, row in zip(latex_names, cases):
        lines.append(
            f"{label} & {row['linear']['peak_front_LTR']:.4f} & "
            f"{row['nonlinear']['peak_front_LTR']:.4f} & "
            f"{row['linear']['peak_rear_LTR']:.4f} & "
            f"{row['nonlinear']['peak_rear_LTR']:.4f}\\\\"
        )
    lines.append(r"\bottomrule\end{tabular}")
    write(TAB / "nonlinear_cases.tex", "\n".join(lines) + "\n")

    coefficients = np.asarray(design["coefficients"]).reshape(2, 6)
    taps = [0, 2, 5, 10, 20, 40]
    coeff_lines = []
    for j, tap in enumerate(taps):
        coeff_lines.append(
            f"{tap} & {0.02*tap:.2f} & {coefficients[0,j]:.12g} & "
            f"{coefficients[1,j]:.12g}\\\\"
        )
    coeff_lines.append(r"\bottomrule")
    write(TAB / "delay_aware_coefficients.tex", "\n".join(coeff_lines) + "\n")

    for name in ("delay_aware_design.png", "nonlinear_critical_cases.png"):
        shutil.copy2(FIG / name, PUB_FIG / name)
    for name in ("delay_aware_summary.tex", "nonlinear_cases.tex",
                 "delay_aware_coefficients.tex"):
        shutil.copy2(TAB / name, PUB_TAB / name)

    manifest = {
        "figures": [str(path.relative_to(ROOT)) for path in sorted(FIG.iterdir())],
        "tables": [str(path.relative_to(ROOT)) for path in sorted(TAB.iterdir())],
    }
    write(OUT / "artifact_manifest.json", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()