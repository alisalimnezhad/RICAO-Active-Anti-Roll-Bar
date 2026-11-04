"""Reproduce and independently audit the main 20 mA result."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from scipy.integrate import solve_ivp

from study import Study, causal_tree_lower, save_json, serial, ROOT, AMP, RATE, H, I0

OUT = Path(__file__).resolve().parent / "outputs"


def hash_inputs():
    paths = sorted((ROOT / "validation").glob("*.py")) + sorted((ROOT / "validation/data").glob("*"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def independent_hold_check(study, f, witness):
    """Compare exact hold recursion with solve_ivp for one adversarial history."""
    tr = study.simulate(f, witness, refine=1)
    exact_nodes = tr["state"]
    wfull = np.r_[0.0, witness]
    x = np.zeros(10)
    maximum = 0.0
    fmat = np.asarray(f).reshape(2, 6)
    from study import DELAYS
    for k in range(study.n):
        maximum = max(maximum, float(np.max(np.abs(x - exact_nodes[k]) / np.asarray([.1,1,.1,1,.1,.1,1e7,1e-3,1e7,1e-3]))))
        feed = np.zeros(2)
        for j, delay in enumerate(DELAYS):
            idx = k - delay
            if idx >= 0:
                feed += fmat[:, j] * wfull[idx]
        u = -study.K @ x + I0 * feed
        d0, d1 = AMP * wfull[k], AMP * wfull[k + 1]
        sol = solve_ivp(
            lambda t, xx: study.A @ xx + study.B @ u + study.D[:, 0] * (d0 + (d1 - d0) * t / H),
            (0.0, H), x, method="Radau", rtol=2e-11, atol=2e-13,
        )
        x = sol.y[:, -1]
    return maximum


def main():
    start = time.time()
    before = hash_inputs()
    study = Study(phases=9)
    f, witnesses, history, design_master, family_master = study.synthesize(verbose=True)
    evaluation = study.evaluate(f, phases=41)
    baseline = study.evaluate(np.zeros(12), phases=41)
    tree = causal_tree_lower(study, fraction=1.0)

    continuous = evaluation["continuous_upper"]
    limits = np.ones(16)
    feasibility_slack = limits[2:] - continuous[2:]
    if np.min(feasibility_slack) < -2e-8:
        raise AssertionError("a certified constraint exceeds its limit")

    worst_phase, worst_output = np.unravel_index(
        np.argmax(evaluation["phase_upper"][:, :2]),
        evaluation["phase_upper"][:, :2].shape,
    )
    witness = evaluation["witnesses"][worst_phase][worst_output]
    trajectory = study.simulate(f, witness, refine=40)
    y = study.output_history(trajectory)
    dense_max = np.max(np.abs(y), axis=0)
    dense_excess = float(np.max(dense_max - continuous))
    if dense_excess > 2e-8:
        raise AssertionError("dense trace exceeds certified bound")

    radau_error = independent_hold_check(study, f, witness)
    after = hash_inputs()
    if before != after:
        raise AssertionError("legacy validation inputs changed")

    old = {
        "proposed_upper": 0.04611481383897718,
        "family_lower": 0.045757890123814454,
        "baseline_upper": 0.15202244835016365,
        "tree_lower": 0.003455678336406757,
        "sample_radius": 0.8757552172268692,
    }
    new = {
        "proposed_upper": float(np.max(continuous[:2])),
        "family_lower": family_master.lower,
        "family_master_primal": family_master.gamma,
        "baseline_upper": float(np.max(baseline["continuous_upper"][:2])),
        "tree_lower": tree["lower"],
        "sample_radius": study.rho_closed,
    }
    comparison = {k: {"old": old[k], "new": new[k],
                       "absolute_difference": abs(new[k] - old[k]),
                       "relative_difference": abs(new[k] - old[k]) / max(abs(old[k]), 1e-30)}
                  for k in old}

    result = {
        "status": "PASS" if max(v["relative_difference"] for v in comparison.values()) < 5e-4 else "REVIEW",
        "scope": {
            "speed_kmh": 70.0, "horizon_s": 4.0, "sample_s": H,
            "steering_amplitude_deg": 2.0, "steering_rate_deg_s": 5.0,
            "current_limit_mA": 20.0, "initial_state": "zero",
        },
        "controller_coefficients": np.asarray(f).reshape(2, 6),
        "delays_samples": [0, 2, 5, 10, 20, 40],
        "delays_seconds": [0.0, 0.04, 0.10, 0.20, 0.40, 0.80],
        "new_results": new,
        "comparison_with_reported_v1": comparison,
        "family_width_percent": 100.0 * (new["proposed_upper"] - new["family_lower"]) / new["proposed_upper"],
        "baseline_upper_reduction_percent": 100.0 * (1.0 - new["proposed_upper"] / new["baseline_upper"]),
        "continuous_output_bounds": dict(zip(study.names, continuous)),
        "grid_output_bounds": dict(zip(study.names, evaluation["grid_upper"])),
        "interpolation_corrections": dict(zip(study.names, evaluation["interpolation_correction"])),
        "minimum_constraint_slack": float(np.min(feasibility_slack)),
        "limiting_constraint": study.names[2 + int(np.argmin(feasibility_slack))],
        "tree_audit": tree,
        "exchange": {
            "iterations": len(history), "witnesses": len(witnesses),
            "final_design_master": serial(design_master.__dict__),
            "physical_family_master": serial(family_master.__dict__),
            "history": history,
        },
        "independent_checks": {
            "dense_trace_max_excess": dense_excess,
            "radau_scaled_state_max_error": radau_error,
            "max_support_dual_correction": float(np.max(evaluation["support_corrections"])),
            "legacy_input_hashes_unchanged": before == after,
            "legacy_input_hashes": after,
        },
        "runtime_seconds": time.time() - start,
        "study_source_sha256": hashlib.sha256((Path(__file__).parent / "study.py").read_bytes()).hexdigest(),
    }
    save_json(OUT / "reproduction_20mA.json", result)
    np.savetxt(OUT / "controller_coefficients_20mA.csv",
               np.column_stack(([0,2,5,10,20,40], [0,.04,.10,.20,.40,.80], np.asarray(f).reshape(2,6).T)),
               delimiter=",", header="delay_samples,delay_seconds,front,rear", comments="")
    np.savez_compressed(OUT / "audit_witness.npz", witness=witness, **trajectory, outputs=y)
    print(json.dumps(serial({k: result[k] for k in ["status", "new_results", "family_width_percent", "baseline_upper_reduction_percent", "minimum_constraint_slack", "limiting_constraint", "independent_checks", "runtime_seconds"]}), indent=2))


if __name__ == "__main__":
    main()