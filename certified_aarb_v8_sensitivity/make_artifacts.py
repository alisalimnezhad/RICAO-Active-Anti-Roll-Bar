"""Create publication tables and figures from the fixed-controller audit."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)


def load():
    return json.loads((OUT / "results.json").read_text())


def display_status(status):
    return {
        "PASS": "Pass",
        "FAIL_CONSTRAINT": "Constraint",
        "FAIL_REAR_DEGRADATION": "Rear degradation",
        "FAIL_UNSTABLE": "Unstable",
    }[status]


def number(value, digits=4):
    return "--" if value is None else f"{float(value):.{digits}f}"


def write_tables(data):
    cases = data["cases"]
    summary_fields = (
        "group", "key", "label", "status", "point_count", "worst_front_LTR",
        "worst_rear_LTR", "minimum_rear_margin_percent",
        "max_constraint_utilization", "max_spectral_radius",
    )
    with (OUT / "case_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({name: case.get(name) for name in summary_fields})

    fir_horizon = []
    for case in cases:
        if case["group"] not in ("fir", "horizon"):
            continue
        variable = (
            str(case["taps"]) if case["group"] == "fir"
            else f"{case['horizon_s']:.0f}"
        )
        group_label = "FIR" if case["group"] == "fir" else "Horizon"
        fir_horizon.append(
            f"{group_label} & {variable} & "
            f"{number(case.get('worst_front_LTR'), 6)} & "
            f"{number(case.get('minimum_rear_margin_percent'), 3)} & "
            f"{number(case.get('max_constraint_utilization'), 3)} & "
            f"{display_status(case['status'])} \\\\"
        )
    (TAB / "fir_horizon.tex").write_text(
        "\n".join(fir_horizon) + "\n\\bottomrule\n"
    )

    delay_lines = [
        f"{row['delay_ms']:.0f} & {row['minimum_spectral_radius']:.3f} & "
        f"{row['maximum_spectral_radius']:.3f} & {row['stable_count']}/170 & "
        f"{display_status(row['status'])} \\\\"
        for row in data["delay"]
    ]
    (TAB / "delay.tex").write_text(
        "\n".join(delay_lines) + "\n\\bottomrule\n"
    )

    hydraulic = [case for case in cases if case["group"] == "hydraulic"]
    parameter_order = ("Kv", "tau", "Kx", "KP", "bulk", "Vt", "area", "Ctp")
    parameter_labels = {
        "Kv": r"$K_v$", "tau": r"$\tau_v$", "Kx": r"$K_x$",
        "KP": r"$K_P$", "bulk": r"$\beta_e$", "Vt": r"$V_t$",
        "area": r"$A_p$", "Ctp": r"$C_{tp}$",
    }
    hydraulic_lines = []
    for parameter in parameter_order:
        rows = [case for case in hydraulic if case.get("parameter") == parameter]
        unstable = [case["label"].split(" ", 1)[1] for case in rows
                    if case["status"] == "FAIL_UNSTABLE"]
        stable = [case for case in rows if case["status"] != "FAIL_UNSTABLE"]
        worst_util = max((case["max_constraint_utilization"] for case in stable),
                         default=None)
        min_margin = min((case["minimum_rear_margin_percent"] for case in stable),
                         default=None)
        max_rho = max(case["max_spectral_radius"] for case in rows)
        unstable_text = ", ".join(unstable).replace("_", r"\_") or "--"
        hydraulic_lines.append(
            f"{parameter_labels[parameter]} & {unstable_text} & {number(worst_util, 3)} & "
            f"{number(min_margin, 2)} & {max_rho:.3f} \\\\"
        )
    (TAB / "hydraulic.tex").write_text(
        "\n".join(hydraulic_lines) + "\n\\bottomrule\n"
    )


def make_fixed_figure(data):
    cases = data["cases"]
    fir = sorted((c for c in cases if c["group"] == "fir"), key=lambda c: c["taps"])
    horizon = sorted((c for c in cases if c["group"] == "horizon"),
                     key=lambda c: c["horizon_s"])
    steering = [c for c in cases if c["group"] == "steering"]
    amps = [1.0, 2.0, 3.0]
    rates = [2.5, 5.0, 7.5]
    util = np.zeros((3, 3))
    margin = np.zeros((3, 3))
    for case in steering:
        i = amps.index(case["amp_deg"])
        j = rates.index(case["rate_deg_s"])
        util[i, j] = case["max_constraint_utilization"]
        margin[i, j] = case["minimum_rear_margin_percent"]

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5,
                         "axes.labelsize": 9, "figure.dpi": 140})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    taps = [c["taps"] for c in fir]
    axes[0, 0].plot(taps, [c["worst_front_LTR"] for c in fir], "o-",
                    label="Worst front LTR")
    axes[0, 0].set_xlabel("Retained taps per channel")
    axes[0, 0].set_ylabel("Worst front LTR")
    axes[0, 0].set_title("(a) FIR truncation")
    axes[0, 0].grid(alpha=.25)
    ax2 = axes[0, 0].twinx()
    ax2.plot(taps, [c["max_constraint_utilization"] for c in fir], "s--",
             color="#D55E00", label="Max. constraint utilization")
    ax2.axhline(1.0, color="black", lw=.8, ls=":")
    ax2.set_ylabel("Max. utilization")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    twin_handles, twin_labels = ax2.get_legend_handles_labels()
    axes[0, 0].legend(handles + twin_handles, labels + twin_labels,
                      loc="upper right", frameon=False, fontsize=6.8)

    horizons = [c["horizon_s"] for c in horizon]
    axes[0, 1].plot(horizons, [c["worst_front_LTR"] for c in horizon], "o-")
    axes[0, 1].set_xlabel("Horizon (s)")
    axes[0, 1].set_ylabel("Worst front LTR")
    axes[0, 1].set_title("(b) Horizon convergence")
    axes[0, 1].grid(alpha=.25)
    axes[0, 1].ticklabel_format(axis="y", style="plain", useOffset=False)

    im0 = axes[1, 0].imshow(util, origin="lower", aspect="auto", cmap="RdYlGn_r",
                            vmin=min(.8, util.min()), vmax=max(1.2, util.max()))
    axes[1, 0].set_xticks(range(3), rates)
    axes[1, 0].set_yticks(range(3), amps)
    axes[1, 0].set_xlabel("Rate limit (deg/s)")
    axes[1, 0].set_ylabel("Amplitude limit (deg)")
    axes[1, 0].set_title("(c) Steering: max. constraint utilization")
    for i in range(3):
        for j in range(3):
            axes[1, 0].text(j, i, f"{util[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im0, ax=axes[1, 0], shrink=.82)

    lim = max(abs(margin.min()), abs(margin.max()))
    im1 = axes[1, 1].imshow(margin, origin="lower", aspect="auto", cmap="RdBu",
                            vmin=-lim, vmax=lim)
    axes[1, 1].set_xticks(range(3), rates)
    axes[1, 1].set_yticks(range(3), amps)
    axes[1, 1].set_xlabel("Rate limit (deg/s)")
    axes[1, 1].set_ylabel("Amplitude limit (deg)")
    axes[1, 1].set_title("(d) Steering: rear margin (%)")
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, f"{margin[i,j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im1, ax=axes[1, 1], shrink=.82)
    fig.savefig(FIG / "fixed_sensitivity_summary.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fixed_sensitivity_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_delay_hydraulic_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    delays = data["delay"]
    x = [row["delay_ms"] for row in delays]
    axes[0].plot(x, [row["maximum_spectral_radius"] for row in delays], "o-",
                 label="Maximum")
    axes[0].plot(x, [row["minimum_spectral_radius"] for row in delays], "s--",
                 label="Minimum")
    axes[0].axhline(1.0, color="black", ls=":", lw=1)
    axes[0].set_xlabel("Common measurement delay (ms)")
    axes[0].set_ylabel("Spectral radius")
    axes[0].set_title("(a) Fixed-loop latency sensitivity")
    axes[0].grid(alpha=.25)
    axes[0].legend(frameon=False)

    hydraulic = [c for c in data["cases"] if c["group"] == "hydraulic"
                 and c.get("parameter") != "nominal"]
    labels = [c["label"].replace("bulk", r"$\beta_e$") for c in hydraulic]
    rho = [c["max_spectral_radius"] for c in hydraulic]
    positions = np.arange(len(hydraulic))
    categories = (
        ("PASS", "Stable / passes", "#0072B2", "o"),
        ("FAIL_STABLE", "Stable / fails", "#E69F00", "^"),
        ("FAIL_UNSTABLE", "Unstable", "#D55E00", "X"),
    )
    for category, label, color, marker in categories:
        indices = [i for i, case in enumerate(hydraulic)
                   if (case["status"] == category if category != "FAIL_STABLE"
                       else case["status"] not in ("PASS", "FAIL_UNSTABLE"))]
        axes[1].scatter(
            positions[indices], np.asarray(rho)[indices], label=label,
            color=color, marker=marker, edgecolors="#222222",
            linewidths=.35, s=30, zorder=3,
        )
    axes[1].axhline(1.0, color="black", ls=":", lw=1)
    axes[1].set_xticks(positions, labels, rotation=90, ha="center", fontsize=6.8)
    axes[1].set_ylabel("Maximum spectral radius")
    axes[1].set_title("(b) Hydraulic perturbations")
    axes[1].grid(axis="y", alpha=.25)
    axes[1].legend(loc="upper right", frameon=False, fontsize=6.5,
                   handletextpad=.35, borderpad=.2)
    fig.savefig(FIG / "delay_hydraulic_summary.pdf", bbox_inches="tight")
    fig.savefig(FIG / "delay_hydraulic_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    data = load()
    write_tables(data)
    make_fixed_figure(data)
    make_delay_hydraulic_figure(data)
    print(json.dumps({"status": "PASS", "figures": 2, "tables": 3}, indent=2))


if __name__ == "__main__":
    main()