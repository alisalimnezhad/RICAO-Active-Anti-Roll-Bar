"""Generate Table 15 and Figure 10 only from solver result JSON."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
TABLE = ROOT / "paper" / "tables" / "table_combined_robustness_summary.tex"
FIGURE = ROOT / "paper" / "figures" / "combined_robustness_summary.png"


def table_text(summaries):
    labels = {"pm5": r"$\pm5\%$", "pm10": r"$\pm10\%$"}
    rows = []
    for bank in ("pm5", "pm10"):
        summary = summaries[bank]
        rows.append(
            f"{labels[bank]} & {summary['sample_count']} & "
            f"{summary['proposed_stable_count']} & {summary['feasible_count']} & "
            f"{summary['rear_pass_count']} & {summary['full_pass_count']} & "
            f"{summary['refined_worst_stable_ltr']:.6f} & "
            f"{summary['refined_max_utilization']:.4f} & "
            f"{summary['refined_min_rear_margin_percent']:.3f} & "
            f"{summary['max_spectral_radius']:.4f} \\\\"
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    result = json.loads((OUT / "results_summary.json").read_text())
    if result.get("status") != "PASS":
        raise RuntimeError("refusing to publish a non-PASS result summary")
    summaries = result["summaries"]
    expected = {"pm5": 512, "pm10": 1024}
    for bank, count in expected.items():
        summary = summaries[bank]
        if not (
            summary["sample_count"] == count
            and summary["finite_count"] == count
            and summary["error_count"] == 0
            and summary["support_success_count"] == summary["proposed_stable_count"]
        ):
            raise RuntimeError(f"incomplete or failed bank: {bank}")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(table_text(summaries))

    gate_names = ["Stable", "Feasible", "Rear cert.", "Full pass"]
    counts = np.array([
        [summaries[bank][key] for key in (
            "proposed_stable_count", "feasible_count", "rear_pass_count",
            "full_pass_count",
        )]
        for bank in ("pm5", "pm10")
    ], dtype=float)
    totals = np.array([512.0, 1024.0])
    colors = ["#2878B5", "#E07B39"]
    names = [r"$\pm5\%$", r"$\pm10\%$"]
    x = np.arange(len(gate_names))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.55), constrained_layout=True)
    for index, (color, name) in enumerate(zip(colors, names)):
        offset = (index - 0.5) * width
        axes[0].bar(x + offset, counts[index], width, color=color, label=name)
        axes[1].bar(
            x + offset, 100.0 * counts[index] / totals[index], width,
            color=color, label=name,
        )
    axes[0].set_ylabel("Number of records")
    axes[0].set_title("(a) Records passing each gate")
    axes[1].set_ylabel(r"Fraction of bank (\%)")
    axes[1].set_title("(b) Gate-passing fraction")
    for axis in axes:
        axis.set_xticks(x, gate_names, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylim(0, 1100)
    axes[1].set_ylim(0, 100)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()