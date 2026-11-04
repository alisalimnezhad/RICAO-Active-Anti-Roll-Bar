"""Independent error budget for the limiting rear-axle certificate.

The stage-4 rear certificate is deliberately one-sided: a feasible sampled
LQR1 value is a lower bound on the baseline peak, whereas a residual-corrected
support value plus the matrix-exponential intersample remainder is an upper
bound on the proposed peak.  This script reconstructs the archived limiting
plant, repeats the calculation on four phase meshes, cross-checks the active
support LPs with three HiGHS algorithms, and writes publication artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V2 = ROOT / "certified_aarb_v2"
V3 = ROOT / "certified_aarb_v3"
V4 = ROOT / "certified_aarb_v4"
OUT = HERE / "outputs" / "error_budget"
OUT.mkdir(parents=True, exist_ok=True)

if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))

from study import ParametricStudy, base, study_at  # noqa: E402


MESHES = (41, 81, 161, 321)
OUTPUT_INDEX = 1
FEASIBILITY_TOL = 5.0e-8
IDENTITY_TOL = 5.0e-12
SOLVER_METHODS = (
    ("highs", None),
    ("highs-ds", {"presolve": False}),
    ("highs-ipm", {"presolve": True}),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve_support(study, vector, method="highs", options=None):
    """Solve one support LP and construct a residual-corrected upper bound."""
    g = np.asarray(vector, dtype=float)
    result = linprog(
        -g,
        A_ub=study.Aw,
        b_ub=study.bw,
        bounds=[(-1.0, 1.0)] * study.n,
        method=method,
        options=options,
    )
    record = {
        "method": method,
        "success": bool(result.success),
        "status_code": int(result.status),
        "message": str(result.message),
    }
    if not result.success:
        return record

    witness = np.asarray(result.x, dtype=float)
    primal = float(g @ witness)
    solver_primal = float(-result.fun)
    dual = np.minimum(np.asarray(result.ineqlin.marginals, dtype=float), 0.0)
    residual = -g - study.Aw.T @ dual
    dual_lower_for_min_problem = float(
        study.bw @ dual - np.abs(residual).sum()
    )
    corrected_upper = float(-dual_lower_for_min_problem)
    rate_violation = float(np.max(study.Aw @ witness - study.bw))
    box_violation = float(max(
        np.max(witness - 1.0), np.max(-1.0 - witness), 0.0
    ))
    record.update({
        "primal_value": primal,
        "solver_primal_value": solver_primal,
        "objective_recompute_error": abs(primal - solver_primal),
        "corrected_upper": corrected_upper,
        "dual_residual_correction": float(corrected_upper - primal),
        "dual_stationarity_residual_inf": float(np.max(np.abs(residual))),
        "rate_constraint_violation": rate_violation,
        "box_constraint_violation": box_violation,
        "witness_feasible": bool(
            rate_violation <= FEASIBILITY_TOL
            and box_violation <= FEASIBILITY_TOL
        ),
        "witness": witness,
    })
    return record


def controller_mesh_record(study, coefficients, phases):
    phase_times = np.linspace(0.0, study.h, int(phases))
    maps = study.maps(phase_times)
    combined = study.combine_maps(maps, np.asarray(coefficients, dtype=float))
    corrected = np.empty(int(phases))
    primal = np.empty(int(phases))
    corrections = np.empty(int(phases))
    witnesses = []
    for phase in range(int(phases)):
        solved = solve_support(study, combined[phase, OUTPUT_INDEX])
        if not solved["success"]:
            raise RuntimeError(solved["message"])
        corrected[phase] = solved["corrected_upper"]
        primal[phase] = solved["primal_value"]
        corrections[phase] = solved["dual_residual_correction"]
        witnesses.append(solved["witness"])

    corrected_phase = int(np.argmax(corrected))
    primal_phase = int(np.argmax(primal))
    spacing = float(study.h / (int(phases) - 1))
    interpolation, second, vbar = study._intersample_correction(
        np.asarray(coefficients, dtype=float), spacing
    )
    remainder = float(interpolation[OUTPUT_INDEX])
    record = {
        "phases": int(phases),
        "phase_spacing_s": spacing,
        "phase_spacing_ms": 1000.0 * spacing,
        "grid_corrected_upper": float(corrected[corrected_phase]),
        "grid_feasible_lower": float(primal[primal_phase]),
        "maximum_phase_correction": float(np.max(corrections)),
        "correction_at_grid_upper": float(corrections[corrected_phase]),
        "corrected_active_phase_index": corrected_phase,
        "corrected_active_phase_time_s": float(phase_times[corrected_phase]),
        "primal_active_phase_index": primal_phase,
        "primal_active_phase_time_s": float(phase_times[primal_phase]),
        "intersample_correction": remainder,
        "continuous_upper": float(corrected[corrected_phase] + remainder),
        "rear_second_derivative_bound": float(second[OUTPUT_INDEX]),
        "interval_start_bound": vbar,
        "corrected_active_vector": combined[corrected_phase, OUTPUT_INDEX],
        "primal_active_vector": combined[primal_phase, OUTPUT_INDEX],
        "corrected_active_witness": witnesses[corrected_phase],
        "primal_active_witness": witnesses[primal_phase],
    }
    study._map_cache.clear()
    return record


def error_budget_row(baseline, proposed):
    primal_gap = (
        baseline["grid_feasible_lower"]
        - proposed["grid_feasible_lower"]
    )
    conservative_dual_budget = (
        proposed["grid_corrected_upper"]
        - proposed["grid_feasible_lower"]
    )
    corrected_sampled_gap = (
        baseline["grid_feasible_lower"]
        - proposed["grid_corrected_upper"]
    )
    remainder = proposed["intersample_correction"]
    certified_margin = corrected_sampled_gap - remainder
    identity_residual = (
        primal_gap - conservative_dual_budget - remainder - certified_margin
    )
    return {
        "phases": proposed["phases"],
        "phase_spacing_ms": proposed["phase_spacing_ms"],
        "baseline_feasible_sampled_lower": baseline["grid_feasible_lower"],
        "baseline_corrected_grid_upper": baseline["grid_corrected_upper"],
        "baseline_max_dual_correction": baseline["maximum_phase_correction"],
        "proposed_feasible_sampled_lower": proposed["grid_feasible_lower"],
        "proposed_corrected_grid_upper": proposed["grid_corrected_upper"],
        "proposed_max_dual_correction": proposed["maximum_phase_correction"],
        "primal_sampled_gap": primal_gap,
        "conservative_dual_residual_budget": conservative_dual_budget,
        "corrected_sampled_gap": corrected_sampled_gap,
        "intersample_remainder": remainder,
        "certified_margin": certified_margin,
        "relative_margin": certified_margin / baseline["grid_feasible_lower"],
        "relative_margin_percent": (
            100.0 * certified_margin / baseline["grid_feasible_lower"]
        ),
        "budget_identity_residual": identity_residual,
    }


def crosscheck_active_supports(study, mesh_records):
    rows = []
    for phases, pair in mesh_records.items():
        active_cases = (
            (
                "baseline_feasible_lower",
                pair["baseline"]["primal_active_vector"],
            ),
            (
                "proposed_corrected_upper",
                pair["proposed"]["corrected_active_vector"],
            ),
        )
        for active_name, vector in active_cases:
            for method, options in SOLVER_METHODS:
                solved = solve_support(study, vector, method, options)
                solved.pop("witness", None)
                rows.append({
                    "phases": int(phases),
                    "active_case": active_name,
                    **solved,
                })
    return rows


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(rows):
    lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"Phases & Spacing (ms) & Sampled gap & Dual budget & Remainder & Margin & Margin (\%)\\",
        r" & & \multicolumn{1}{c}{$10^{-5}$} & \multicolumn{1}{c}{$10^{-8}$} & \multicolumn{1}{c}{$10^{-5}$} & \multicolumn{1}{c}{$10^{-5}$} & \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['phases']} & {row['phase_spacing_ms']:.4f} & "
            f"{1e5 * row['primal_sampled_gap']:.4f} & "
            f"{1e8 * row['conservative_dual_residual_budget']:.4f} & "
            f"{1e5 * row['intersample_remainder']:.4f} & "
            f"{1e5 * row['certified_margin']:.4f} & "
            f"{row['relative_margin_percent']:.3f}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "error_budget_table.tex").write_text("\n".join(lines) + "\n")


def make_figure(rows):
    phases = np.asarray([row["phases"] for row in rows])
    relative_margin = np.asarray([
        row["relative_margin_percent"] for row in rows
    ])
    sampled_gap = np.asarray([row["primal_sampled_gap"] for row in rows])
    dual = np.asarray([
        row["conservative_dual_residual_budget"] for row in rows
    ])
    remainder = np.asarray([row["intersample_remainder"] for row in rows])

    with plt.rc_context({
        "font.size": 10.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.7,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.35))
        axes[0].axhline(0.0, color="black", linewidth=0.9, linestyle="--")
        axes[0].plot(
            phases, relative_margin, "o-", linewidth=1.8, markersize=5.5,
            color="#006271", label="certified relative margin",
        )
        for phase, value in zip(phases, relative_margin):
            offset = (0, 8) if value >= 0.0 else (0, -15)
            axes[0].annotate(
                f"{value:.3f}%", (phase, value), textcoords="offset points",
                xytext=offset, ha="center", fontsize=8.2,
            )
        axes[0].annotate(
            "coarse bound\ninconclusive", (phases[0], relative_margin[0]),
            textcoords="offset points", xytext=(28, -3), ha="left",
            va="center", fontsize=8.2, color="#9a2f23",
            arrowprops={"arrowstyle": "->", "color": "#9a2f23", "lw": 0.8},
        )
        axes[0].set_title("(a) Certified margin under mesh refinement")
        axes[0].set_xlabel("Phases per 20 ms interval")
        axes[0].set_ylabel("Rear-axle margin relative to LQR1 (%)")
        axes[0].set_xticks(phases)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(frameon=False, loc="lower right")

        axes[1].semilogy(
            phases, sampled_gap, "^-", linewidth=1.7, markersize=5.5,
            label="primal sampled gap",
        )
        axes[1].semilogy(
            phases, remainder, "o-", linewidth=1.7, markersize=5.5,
            label="intersample remainder",
        )
        axes[1].semilogy(
            phases, np.maximum(dual, 1e-18), "s-", linewidth=1.7,
            markersize=5.0, label="residual-dual budget",
        )
        axes[1].set_title("(b) Components of the one-sided certificate")
        axes[1].set_xlabel("Phases per 20 ms interval")
        axes[1].set_ylabel("Absolute contribution")
        axes[1].set_xticks(phases)
        axes[1].grid(True, which="both", alpha=0.25)
        axes[1].legend(frameon=False, loc="center right")
        fig.tight_layout(pad=0.7, w_pad=1.4)
        fig.savefig(
            OUT / "rear_error_budget_mesh.pdf", bbox_inches="tight",
            facecolor="white",
        )
        fig.savefig(
            OUT / "rear_error_budget_mesh.png", dpi=320,
            bbox_inches="tight", facecolor="white",
        )
        plt.close(fig)


def point_key(row):
    return (
        round(float(row["speed_kmh"]), 12),
        round(float(row["mass_fraction"]), 12),
        round(float(row["payload_height"]), 12),
    )


def main():
    start = time.time()
    verification_path = V4 / "outputs" / "envelope_verification.json"
    screening_path = V4 / "outputs" / "envelope_results.json"
    coefficient_path = V4 / "outputs" / "fixed_rear_protected_coefficients.csv"
    archived = json.loads(verification_path.read_text())
    screening = json.loads(screening_path.read_text())
    limiting = archived["limiting_rear_point"]
    limiting_key = point_key(limiting)
    matching_screen_rows = [
        row for row in screening["rows"] if point_key(row) == limiting_key
    ]
    if len(matching_screen_rows) != 1:
        raise RuntimeError("limiting point not uniquely present in archived bank")

    nominal = ParametricStudy(speed_kmh=70.0)
    fixed_gain = nominal.K.copy()
    study = study_at(
        limiting["speed_kmh"],
        limiting["mass_fraction"],
        limiting["payload_height"],
        fixed_gain,
        phases=9,
        label="archived limiting rear point",
    )
    baseline_coefficients = np.zeros(12)
    proposed_coefficients = np.loadtxt(
        coefficient_path, delimiter=","
    ).reshape(-1)

    mesh_records = {}
    budget_rows = []
    for phases in MESHES:
        print(f"mesh {phases}: baseline", flush=True)
        baseline = controller_mesh_record(
            study, baseline_coefficients, phases
        )
        print(f"mesh {phases}: proposed", flush=True)
        proposed = controller_mesh_record(
            study, proposed_coefficients, phases
        )
        row = error_budget_row(baseline, proposed)
        if abs(row["budget_identity_residual"]) > IDENTITY_TOL:
            raise RuntimeError("error-budget identity failed")
        mesh_records[str(phases)] = {
            "baseline": baseline,
            "proposed": proposed,
            "budget": row,
        }
        budget_rows.append(row)

    print("cross-checking active support LPs", flush=True)
    solver_rows = crosscheck_active_supports(study, mesh_records)
    successful_solver_rows = [row for row in solver_rows if row["success"]]
    all_solvers_successful = len(successful_solver_rows) == len(solver_rows)
    all_witnesses_feasible = all(
        row.get("witness_feasible", False) for row in solver_rows
    )
    method_spreads = []
    for phases in MESHES:
        for active_case in (
            "baseline_feasible_lower", "proposed_corrected_upper"
        ):
            group = [
                row for row in solver_rows
                if row["phases"] == phases
                and row["active_case"] == active_case
                and row["success"]
            ]
            corrected = [row["corrected_upper"] for row in group]
            primal = [row["primal_value"] for row in group]
            method_spreads.append({
                "phases": phases,
                "active_case": active_case,
                "corrected_upper_spread": float(max(corrected) - min(corrected)),
                "primal_value_spread": float(max(primal) - min(primal)),
            })

    finest = mesh_records[str(MESHES[-1])]
    archived_reconstruction = {
        "baseline_feasible_lower_difference_81": float(
            mesh_records["81"]["baseline"]["grid_feasible_lower"]
            - limiting["baseline"]["grid_feasible_lower"]
        ),
        "proposed_corrected_upper_difference_81": float(
            mesh_records["81"]["proposed"]["grid_corrected_upper"]
            - limiting["proposed"]["grid_corrected_upper"]
        ),
        "proposed_remainder_difference_81": float(
            mesh_records["81"]["proposed"]["intersample_correction"]
            - limiting["proposed"]["intersample_correction"]
        ),
        "certified_margin_difference_81": float(
            mesh_records["81"]["budget"]["certified_margin"]
            - limiting["certified_margin"]
        ),
    }
    reconstruction_ok = max(
        abs(value) for value in archived_reconstruction.values()
    ) <= 5e-12
    mesh_all_positive = all(row["certified_margin"] > 0.0 for row in budget_rows)
    certification_meshes_positive = all(
        row["certified_margin"] > 0.0
        for row in budget_rows if row["phases"] >= 81
    )
    max_solver_corrected_spread = max(
        row["corrected_upper_spread"] for row in method_spreads
    )
    max_solver_primal_spread = max(
        row["primal_value_spread"] for row in method_spreads
    )
    finest_margin_dominates_solver_spread = (
        finest["budget"]["certified_margin"]
        > 100.0 * max_solver_corrected_spread
    )
    status = "PASS" if all((
        reconstruction_ok,
        certification_meshes_positive,
        all_solvers_successful,
        all_witnesses_feasible,
        max_solver_corrected_spread <= 5e-8,
        max_solver_primal_spread <= 5e-8,
        finest_margin_dominates_solver_spread,
    )) else "REVIEW"

    source_paths = {
        "run_error_budget.py": Path(__file__),
        "v4_verify_envelope.py": V4 / "verify_envelope.py",
        "v4_study.py": V4 / "study.py",
        "v3_study.py": V3 / "study.py",
        "v2_study.py": V2 / "study.py",
        "vehicle_model.py": ROOT / "validation" / "vehicle_model.py",
        "parameters.py": ROOT / "validation" / "parameters.py",
        "envelope_verification.json": verification_path,
        "envelope_results.json": screening_path,
        "fixed_rear_protected_coefficients.csv": coefficient_path,
    }
    result = {
        "status": status,
        "date": "2026-09-23",
        "scope": (
            "Independent numerical error budget at the archived limiting "
            "rear-axle point; it is not a plant-uncertainty or hardware margin."
        ),
        "certificate_definition": (
            "baseline feasible sampled lower minus proposed residual-corrected "
            "grid upper minus proposed intersample remainder"
        ),
        "limiting_point": {
            "set": limiting["set"],
            "speed_kmh": limiting["speed_kmh"],
            "mass_fraction": limiting["mass_fraction"],
            "payload_height": limiting["payload_height"],
        },
        "closed_loop_spectral_radius": float(study.rho_closed),
        "meshes": mesh_records,
        "mesh_summary": budget_rows,
        "finest_mesh": {
            "phases": MESHES[-1],
            **finest["budget"],
            "baseline": finest["baseline"],
            "proposed": finest["proposed"],
        },
        "solver_crosscheck": solver_rows,
        "solver_method_spreads": method_spreads,
        "checks": {
            "archived_81_phase_reconstruction": reconstruction_ok,
            "archived_reconstruction_differences": archived_reconstruction,
            "all_mesh_margins_positive": mesh_all_positive,
            "certification_meshes_81_and_finer_positive": certification_meshes_positive,
            "coarse_41_phase_bound_inconclusive": bool(
                mesh_records["41"]["budget"]["certified_margin"] <= 0.0
            ),
            "all_solver_runs_successful": all_solvers_successful,
            "all_active_witnesses_feasible": all_witnesses_feasible,
            "max_solver_corrected_upper_spread": max_solver_corrected_spread,
            "max_solver_primal_value_spread": max_solver_primal_spread,
            "finest_margin_to_solver_spread_ratio": float(
                finest["budget"]["certified_margin"]
                / max_solver_corrected_spread
            ),
            "finest_margin_dominates_solver_spread": finest_margin_dominates_solver_spread,
            "maximum_budget_identity_residual": max(
                abs(row["budget_identity_residual"]) for row in budget_rows
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "source_sha256": {
            name: sha256(path) for name, path in source_paths.items()
        },
        "runtime_seconds": float(time.time() - start),
    }
    base.save_json(OUT / "error_budget.json", result)

    budget_fields = list(budget_rows[0].keys())
    write_csv(OUT / "mesh_audit.csv", budget_rows, budget_fields)
    solver_fields = [key for key in solver_rows[0] if key != "witness"]
    write_csv(OUT / "solver_crosscheck.csv", solver_rows, solver_fields)
    write_latex_table(budget_rows)
    make_figure(budget_rows)
    np.savez_compressed(
        OUT / "active_witnesses_321.npz",
        baseline=finest["baseline"]["primal_active_witness"],
        proposed=finest["proposed"]["corrected_active_witness"],
        proposed_coefficients=proposed_coefficients,
    )

    artifacts = [
        "error_budget.json",
        "mesh_audit.csv",
        "solver_crosscheck.csv",
        "error_budget_table.tex",
        "rear_error_budget_mesh.pdf",
        "rear_error_budget_mesh.png",
        "active_witnesses_321.npz",
    ]
    manifest = {
        "status": status,
        "date": "2026-09-23",
        "artifacts": {
            name: sha256(OUT / name) for name in artifacts
        },
    }
    base.save_json(OUT / "artifact_manifest.json", manifest)

    summary = {
        "status": status,
        "limiting_point": result["limiting_point"],
        "archived_margin_81": mesh_records["81"]["budget"]["certified_margin"],
        "finest_margin_321": finest["budget"]["certified_margin"],
        "finest_relative_margin_percent": finest["budget"]["relative_margin_percent"],
        "finest_dual_budget": finest["budget"]["conservative_dual_residual_budget"],
        "finest_intersample_remainder": finest["budget"]["intersample_remainder"],
        "max_solver_corrected_upper_spread": max_solver_corrected_spread,
        "all_active_witnesses_feasible": all_witnesses_feasible,
        "runtime_seconds": result["runtime_seconds"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()