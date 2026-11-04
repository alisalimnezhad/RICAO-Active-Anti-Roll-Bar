"""Generate the architecture, certificate-scope and complexity artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import timeit

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
for path in (OUT, FIG, TAB):
    path.mkdir(parents=True, exist_ok=True)
PUB_FIG = ROOT / "paper" / "figures"
PUB_TAB = ROOT / "paper" / "tables"
PUB_FIG.mkdir(parents=True, exist_ok=True)
PUB_TAB.mkdir(parents=True, exist_ok=True)

V4_RESULTS = ROOT / "certified_aarb_v4" / "outputs" / "envelope_results.json"
RESERVE = ROOT / "certified_aarb_v5_rear_audit" / "outputs" / "comparison" / "reserve_comparison.json"
SENSITIVITY = ROOT / "certified_aarb_v8_sensitivity" / "outputs" / "results.json"
DELAY = ROOT / "certified_aarb_v9_delay_aware" / "outputs" / "results.json"
DELAY_GAIN = ROOT / "certified_aarb_v9_delay_aware" / "outputs" / "gain_scan.json"
DELAY_DESIGN = ROOT / "certified_aarb_v9_delay_aware" / "outputs" / "design.json"
V4_DESIGN = ROOT / "certified_aarb_v4" / "outputs" / "fixed_rear_protected_design.json"
RESERVE_DESIGN = ROOT / "certified_aarb_v5_rear_audit" / "outputs" / "reserve_5pct" / "design.json"
SCHEDULED = [
    ROOT / "certified_aarb_v3" / "outputs" / f"bank_scheduled_{node}.json"
    for node in ("50p0", "70p0", "75p0")
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def box(ax, xy, wh, text, color, fontsize=8.3, dashed=False):
    x, y = xy; w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2, edgecolor=color,
        facecolor=(*plt.matplotlib.colors.to_rgb(color), 0.10),
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center",
            fontsize=fontsize, color="#152238")
    return patch


def arrow(ax, start, end, color="#334155", dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.15,
        color=color, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
    ))


def architecture_figure():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.015, 0.56), 0.97, 0.40,
                 boxstyle="round,pad=.01", facecolor="#EFF6FF",
                 edgecolor="#2563EB", linewidth=1.4))
    ax.add_patch(FancyBboxPatch((0.015, 0.08), 0.97, 0.39,
                 boxstyle="round,pad=.01", facecolor="#F0FDFA",
                 edgecolor="#0F766E", linewidth=1.4))
    ax.text(.035, .925, "OFFLINE DESIGN AND VERIFICATION", weight="bold",
            fontsize=11, color="#1D4ED8")
    ax.text(.035, .435, "ONLINE IMPLEMENTATION (20 ms update)", weight="bold",
            fontsize=11, color="#0F766E")

    xs = [.035, .225, .415, .605, .795]
    texts = [
        "Declared plant bank\n10-state linear model\n42 structured\n+ 128 Sobol",
        "Exact lifting\nsteering FOH\ncurrent ZOH\n$T=4$ s, $h_s=20$ ms",
        "Support LP\n+ exchange design\nworst witnesses\nFIR / gain / schedule",
        "Independent audit\n81--321 phases\nresidual correction\n+ intersample budget",
        "Versioned artifacts\nfull-precision coefficients\nJSON/CSV\ntests and figures",
    ]
    colors = ["#2563EB", "#3B82F6", "#7C3AED", "#9333EA", "#4F46E5"]
    for x, text, color in zip(xs, texts, colors):
        box(ax, (x, .65), (.165, .19), text, color, fontsize=7.15)
    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, (left+.165, .745), (right, .745), "#475569")

    online_x = [.035, .19, .345, .505, .665, .82]
    online_text = [
        "state $x$ and\nsteering $\\delta$",
        "optional speed\nschedule interpolation",
        "feedback $-Kx$\n+ causal FIR",
        "current guard\n$|u_i|\\leq20$ mA",
        "ZOH hydraulic\nAARB actuators",
        "vehicle plant\nand monitored outputs",
    ]
    online_colors = ["#0F766E", "#0891B2", "#059669", "#D97706", "#0D9488", "#047857"]
    for x, text, color in zip(online_x, online_text, online_colors):
        box(ax, (x, .225), (.125, .13), text, color, fontsize=8.1)
    for left, right in zip(online_x[:-1], online_x[1:]):
        arrow(ax, (left+.125, .29), (right, .29), "#334155")
    arrow(ax, (.945, .225), (.095, .205), "#0F766E", rad=-.22)
    ax.text(.50, .105,
            "FIR taps: 0, 2, 5, 10, 20 and 40 samples  |  separate 5 ms delay-aware branch",
            ha="center", va="center", fontsize=8.3, color="#0F766E")

    box(ax, (.73, .49), (.24, .075),
        "Nonlinear critical-case screen\n(outside the linear certificate)",
        "#DC2626", fontsize=8.0, dashed=True)
    arrow(ax, (.79, .65), (.80, .565), "#DC2626", dashed=True)
    fig.savefig(FIG / "certification_architecture.pdf", bbox_inches="tight")
    fig.savefig(FIG / "certification_architecture.png", dpi=240,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def scope_table():
    rows = [
        ("Plant dynamics", "Published ten-state linear vehicle--hydraulic model",
         "Nonlinear tyre/wheel-lift or hardware certificate"),
        ("Initial condition", "$x(0)=0$", "Nonzero or uncertain initial states"),
        ("Steering family", r"FOH, $|\delta|\le2^\circ$, $|\dot\delta|\le5^\circ$/s, $w_0=0$",
         "Road inputs and unrestricted driver manoeuvres"),
        ("Time", r"$T=4$ s; $h_s=20$ ms; continuous-time remainder",
         "Infinite-horizon or arbitrary sampling guarantees"),
        ("Actuator/model limits", "Current, force, spool, flow, suspension roll, body/sideslip and tyre slip",
         "Pump power, temperature, cavitation and wear"),
        ("Vehicle parameters", "42 structured + 128 scrambled-Sobol speed--payload plants",
         "Interval proof between the 170 evaluated points"),
        ("Measurements", "Full state and current steering; separate known 5 ms delayed design",
         "Observer error, noise and uncertain/time-varying delay"),
        ("Numerical claim", "Residual-corrected support LP plus intersample bound at each plant",
         "Outward-rounded interval arithmetic or symbolic proof"),
        ("Rear comparison", "One-sided proposed upper versus feasible comparator lower",
         "Hardware robustness margin"),
        ("Additional evidence", "Reference reproduction and five nonlinear critical-case screens",
         "Experiment, HiL or commercial multibody validation"),
    ]
    lines = []
    for item, included, excluded in rows:
        lines.append(f"{item} & {included} & {excluded} \\\\")
    lines.append(r"\bottomrule")
    (TAB / "certificate_scope.tex").write_text("\n".join(lines) + "\n")
    return rows


def benchmark_policy(kind, K, F, loops=30000):
    rng = np.random.default_rng(20260923)
    x = rng.standard_normal(10)
    h6 = rng.standard_normal(6)
    h4 = h6[:4]
    if kind == "lqr":
        statement = "K @ x"
        globals_ = {"K": K, "x": x}
    elif kind == "fir4":
        statement = "K @ x + F @ h4"
        globals_ = {"K": K, "x": x, "F": F[:, :4], "h4": h4}
    elif kind == "fixed":
        statement = "K @ x + F @ h6"
        globals_ = {"K": K, "x": x, "F": F, "h6": h6}
    elif kind == "scheduled":
        K2 = 1.03 * K; F2 = .97 * F; a = .37
        statement = "((1-a)*K+a*K2) @ x + ((1-a)*F+a*F2) @ h6"
        globals_ = {"K": K, "K2": K2, "F": F, "F2": F2,
                    "x": x, "h6": h6, "a": a}
    else:
        raise ValueError(kind)
    samples = timeit.repeat(statement, globals=globals_, number=loops, repeat=7)
    return 1e6 * float(np.median(samples)) / loops


def controller_records():
    v4 = json.loads(V4_RESULTS.read_text())
    reserve = json.loads(RESERVE.read_text())
    sensitivity = json.loads(SENSITIVITY.read_text())
    delay = json.loads(DELAY.read_text())["summary"]
    delay_gain = json.loads(DELAY_GAIN.read_text())
    delay_design = json.loads(DELAY_DESIGN.read_text())
    v4_design = json.loads(V4_DESIGN.read_text())
    reserve_design = json.loads(RESERVE_DESIGN.read_text())
    scheduled_designs = [json.loads(path.read_text()) for path in SCHEDULED]

    sys.path.insert(0, str(ROOT / "certified_aarb_v4"))
    from study import ParametricStudy  # noqa: E402
    K = ParametricStudy(speed_kmh=70.0).K
    F = np.loadtxt(
        ROOT / "certified_aarb_v5_rear_audit" / "outputs" /
        "reserve_5pct" / "coefficients.csv", delimiter=","
    ).reshape(2, 6)
    fir4 = next(case for case in sensitivity["cases"] if case["key"] == "fir_4")
    selected = next(row for row in reserve["designs"] if row["key"] == "reserve_5pct")
    rows = v4["rows"]
    baseline_util = max(row["baseline"]["constraint_utilization"] for row in rows)
    scheduled_runtime = sum(float(row["runtime_seconds"]) for row in scheduled_designs)
    timing = {
        "lqr": benchmark_policy("lqr", K, F),
        "fir4": benchmark_policy("fir4", K, F),
        "fixed": benchmark_policy("fixed", K, F),
        "scheduled": benchmark_policy("scheduled", K, F),
    }
    records = [
        {"controller": "Sampled LQR1", "gain": 20, "fir": 0, "stored": 20,
         "macs": "20", "online_us": timing["lqr"], "schedule": "No",
         "coverage": "170", "worst_ltr": v4["baseline_worst_LTR"],
         "utilization": baseline_util, "offline_s": None,
         "rear": "No axle-specific proof", "util_note": "selected outputs"},
        {"controller": "Four-tap reserve", "gain": 20, "fir": 8, "stored": 28,
         "macs": "28", "online_us": timing["fir4"], "schedule": "No",
         "coverage": "42 structured", "worst_ltr": fir4["worst_LTR"],
         "utilization": fir4["max_constraint_utilization"], "offline_s": None,
         "rear": f"{fir4['minimum_rear_margin_percent']:.2f}% screen",
         "util_note": "all declared"},
        {"controller": "Fixed balanced", "gain": 20, "fir": 12, "stored": 32,
         "macs": "32", "online_us": timing["fixed"], "schedule": "No",
         "coverage": "170", "worst_ltr": v4["controllers"]["fixed_balanced"]["worst_LTR"],
         "utilization": v4["controllers"]["fixed_balanced"]["max_constraint_utilization"],
         "offline_s": None, "rear": "No", "util_note": "all declared"},
        {"controller": "Rear protected", "gain": 20, "fir": 12, "stored": 32,
         "macs": "32", "online_us": timing["fixed"], "schedule": "No",
         "coverage": "170", "worst_ltr": v4["controllers"]["fixed_rear_protected"]["worst_LTR"],
         "utilization": v4["controllers"]["fixed_rear_protected"]["max_constraint_utilization"],
         "offline_s": float(v4_design["runtime_seconds"]), "rear": "0.336% min.",
         "util_note": "all declared"},
        {"controller": "Selected 5% reserve", "gain": 20, "fir": 12, "stored": 32,
         "macs": "32", "online_us": timing["fixed"], "schedule": "No",
         "coverage": "170", "worst_ltr": selected["worst_front_bound"],
         "utilization": selected["max_constraint_utilization"],
         "offline_s": float(reserve_design["runtime_seconds"]),
         "rear": f"{selected['achieved_minimum_margin_percent']:.2f}% min.",
         "util_note": "all declared"},
        {"controller": "Scheduled balanced", "gain": 60, "fir": 36, "stored": 96,
         "macs": "32+64 interp.", "online_us": timing["scheduled"], "schedule": "Speed",
         "coverage": "170", "worst_ltr": v4["controllers"]["scheduled_balanced"]["worst_LTR"],
         "utilization": v4["controllers"]["scheduled_balanced"]["max_constraint_utilization"],
         "offline_s": scheduled_runtime, "rear": "No", "util_note": "all declared"},
        {"controller": "5 ms delay-aware", "gain": 20, "fir": 12, "stored": 32,
         "macs": "32", "online_us": timing["fixed"], "schedule": "No",
         "coverage": "170", "worst_ltr": delay["worst_LTR"],
         "utilization": delay["max_constraint_utilization"],
         "offline_s": float(delay_gain["runtime_seconds"])+float(delay_design["runtime_seconds"]),
         "rear": f"{delay['minimum_rear_margin_percent_vs_delayed_feedback_only']:.1f}% vs delayed FB",
         "util_note": "all declared"},
    ]
    return records, timing


def complexity_tables(records):
    implementation = []
    performance = []
    for row in records:
        controller = row["controller"].replace("%", r"\%")
        schedule = row["schedule"].replace("%", r"\%")
        coverage = row["coverage"].replace("%", r"\%")
        rear = row["rear"].replace("%", r"\%")
        implementation.append(
            f"{controller} & {row['gain']} & {row['fir']} & {row['stored']} & "
            f"{row['macs']} & {row['online_us']:.2f} & {schedule} \\\\")
        offline = "--" if row["offline_s"] is None else f"{row['offline_s']:.0f}"
        performance.append(
            f"{controller} & {coverage} & {row['worst_ltr']:.5f} & "
            f"{row['utilization']:.3f} & {rear} & {offline} \\\\")
    implementation.append(r"\bottomrule")
    performance.append(r"\bottomrule")
    (TAB / "complexity_implementation.tex").write_text("\n".join(implementation)+"\n")
    (TAB / "complexity_performance.tex").write_text("\n".join(performance)+"\n")


def main():
    architecture_figure()
    scope = scope_table()
    records, timing = controller_records()
    complexity_tables(records)
    payload = {
        "date": "2026-09-23",
        "status": "PASS",
        "scope_rows": [
            {"dimension": a, "included": b, "excluded": c}
            for a, b, c in scope
        ],
        "controllers": records,
        "benchmark": {
            "method": "median of seven timeit repeats, 30000 calls each",
            "environment": "Python/NumPy host microbenchmark; not ECU timing",
            "microseconds_per_call": timing,
        },
        "definitions": {
            "stored_scalars": "feedback plus FIR coefficients retained online",
            "macs": "coefficient products per control update; schedule interpolation shown separately",
            "offline_seconds": "archived wall time when a directly attributable design record exists",
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (V4_RESULTS, RESERVE, SENSITIVITY, DELAY, DELAY_GAIN,
                         DELAY_DESIGN, V4_DESIGN, RESERVE_DESIGN, *SCHEDULED)
        },
    }
    save_json(OUT / "presentation_data.json", payload)
    for name in ("certification_architecture.pdf", "certification_architecture.png"):
        shutil.copy2(FIG / name, PUB_FIG / name)
    for name in ("certificate_scope.tex", "complexity_implementation.tex",
                 "complexity_performance.tex"):
        shutil.copy2(TAB / name, PUB_TAB / name)
    artifacts = {}
    # The verifier rewrites verification.json, so neither that self-report nor
    # the manifest itself belongs in the immutable artifact hash set.
    for path in sorted(item for item in OUT.rglob("*") if item.is_file()
                       and item.name not in {"artifact_manifest.json", "verification.json"}):
        artifacts[str(path.relative_to(HERE))] = sha256(path)
    save_json(OUT / "artifact_manifest.json", artifacts)
    print(json.dumps({"status": "PASS", "scope_rows": len(scope),
                      "controllers": len(records)}, indent=2))


if __name__ == "__main__":
    main()