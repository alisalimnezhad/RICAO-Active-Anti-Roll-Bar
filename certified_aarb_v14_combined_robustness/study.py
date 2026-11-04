"""Combined plant/controller stress audit used by manuscript version 14.

The validated stage-4 plant and support-LP implementation are reused without
changing their equations.  This module adds the predeclared 50-dimensional
Sobol perturbation map and the finite/stability/feasibility/rear gates.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V4_STUDY = ROOT / "certified_aarb_v4" / "study.py"
_spec = importlib.util.spec_from_file_location("v14_stage4_study", V4_STUDY)
v4 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = v4
_spec.loader.exec_module(v4)

ParametricStudy = v4.ParametricStudy
distributed_payload_overrides = v4.distributed_payload_overrides
p = v4.p

STRESS_DIMENSION = 50
BANKS = (
    {"name": "pm5", "radius": 0.05, "power": 9, "seed": 2026092705},
    {"name": "pm10", "radius": 0.10, "power": 10, "seed": 2026092710},
)
PLANT_FIELDS = (
    "Cf", "Cr", "kf", "kr", "bf", "br", "ktf", "ktr",
    "area", "Vt", "bulk", "Kx", "KP", "tau", "Kv",
)
PLANT_LABELS = (
    "C_F", "C_R", "k_F", "k_R", "b_F", "b_R", "k_tF", "k_tR",
    "A_p", "V_t", "beta_e", "K_x", "K_P", "tau_v", "K_v",
)


@lru_cache(maxsize=1)
def nominal_assets() -> tuple[np.ndarray, np.ndarray]:
    """Return fixed nominal LQR1 and the selected 5% rear-reserve FIR."""
    gain = ParametricStudy(speed_kmh=70.0).K.copy()
    coefficient_file = (
        ROOT / "certified_aarb_v5_rear_audit" / "outputs" /
        "reserve_5pct" / "coefficients.csv"
    )
    if not coefficient_file.exists():
        raise FileNotFoundError(
            "missing selected FIR coefficients: " + str(coefficient_file)
        )
    fir = np.loadtxt(coefficient_file, delimiter=",").reshape(2, 6)
    return gain, fir


def perturbation_map(unit: np.ndarray, radius: float) -> dict:
    """Map a 50-dimensional Sobol point to physical/model quantities."""
    unit = np.asarray(unit, dtype=float)
    if unit.shape != (STRESS_DIMENSION,):
        raise ValueError(f"expected {STRESS_DIMENSION} Sobol coordinates")
    if not (np.isfinite(unit).all() and np.all(unit >= 0.0) and np.all(unit < 1.0)):
        raise ValueError("Sobol coordinates must be finite and lie in [0,1)")
    radius = float(radius)
    if radius not in (0.05, 0.10):
        raise ValueError("stress radius must be 0.05 or 0.10")

    speed_kmh = 50.0 + 25.0 * unit[0]
    mass_fraction = 0.10 * unit[1]
    payload_height = float(p.h) + (1.80 - float(p.h)) * unit[2]
    overrides = distributed_payload_overrides(mass_fraction, payload_height)

    factors = 1.0 + radius * (2.0 * unit[3:] - 1.0)
    plant_factors = factors[:15]
    gain_factors = factors[15:35].reshape(2, 10)
    fir_factors = factors[35:47].reshape(2, 6)
    if factors.shape != (47,):
        raise AssertionError("incorrect uncertainty-coordinate partition")
    for field, factor in zip(PLANT_FIELDS, plant_factors):
        overrides[field] = float(getattr(p, field) * factor)

    nominal_gain, nominal_fir = nominal_assets()
    return {
        "speed_kmh": float(speed_kmh),
        "mass_fraction": float(mass_fraction),
        "payload_height": float(payload_height),
        "overrides": overrides,
        "plant_multipliers": dict(zip(PLANT_LABELS, map(float, plant_factors))),
        "gain": nominal_gain * gain_factors,
        "fir": nominal_fir * fir_factors,
        "baseline_gain": nominal_gain,
        "gain_multipliers": gain_factors,
        "fir_multipliers": fir_factors,
    }


def finite_study(study: ParametricStudy) -> bool:
    arrays = (
        study.A, study.B, study.D, study.K, study.ad, study.bd_phys,
        study.g0, study.g1, study.ac, study.C, study.Du, study.Ddelta,
        study.Aw, study.bw,
    )
    return all(np.isfinite(array).all() for array in arrays) and np.isfinite(
        study.rho_closed
    )


def evaluate_all_outputs(study: ParametricStudy, fir: np.ndarray, phases: int) -> dict:
    """Solve all signed support LPs and add the intersample remainder."""
    result = study.evaluate(np.asarray(fir).reshape(-1), phases=int(phases))
    continuous = np.asarray(result["continuous_upper"], dtype=float)
    if not np.isfinite(continuous).all():
        raise RuntimeError("non-finite continuous output bound")
    non_ltr = continuous[2:]
    limiting_local = int(np.argmax(non_ltr))
    return {
        "continuous": dict(zip(study.names, map(float, continuous))),
        "worst_ltr": float(np.max(continuous[:2])),
        "max_utilization": float(non_ltr[limiting_local]),
        "limiting_constraint": study.names[limiting_local + 2],
        "max_support_correction": float(np.max(result["support_corrections"])),
    }


def rear_certificate(
    proposed: ParametricStudy,
    baseline: ParametricStudy,
    fir: np.ndarray,
    phases: int,
) -> dict:
    """One-sided proposed-upper versus baseline-feasible-lower rear test."""
    phase_times = np.linspace(0.0, proposed.h, int(phases))
    proposed_g = proposed.combine_maps(
        proposed.maps(phase_times), np.asarray(fir).reshape(-1)
    )
    baseline_g = baseline.combine_maps(
        baseline.maps(phase_times), np.zeros(12)
    )
    proposed_corrected = []
    baseline_feasible = []
    max_support_correction = 0.0
    for phase in range(int(phases)):
        upper, _, _, correction = proposed.support(
            proposed_g[phase, 1], return_witness=True
        )
        proposed_corrected.append(float(upper))
        max_support_correction = max(max_support_correction, float(correction))
        _, _, feasible, _ = baseline.support(
            baseline_g[phase, 1], return_witness=True
        )
        baseline_feasible.append(float(feasible))

    spacing = proposed.h / (int(phases) - 1)
    interpolation, _, _ = proposed._intersample_correction(
        np.asarray(fir).reshape(-1), spacing
    )
    proposed_upper = max(proposed_corrected) + float(interpolation[1])
    baseline_lower = max(baseline_feasible)
    margin = baseline_lower - proposed_upper
    relative = margin / baseline_lower
    return {
        "baseline_feasible_lower": float(baseline_lower),
        "proposed_continuous_upper": float(proposed_upper),
        "certified_margin": float(margin),
        "relative_margin": float(relative),
        "relative_margin_percent": float(100.0 * relative),
        "max_support_correction": float(max_support_correction),
    }


def evaluate_record(
    record_id: str,
    unit: np.ndarray,
    radius: float,
    output_phases: int = 21,
    rear_phases: int = 41,
) -> dict:
    """Evaluate one record with finite and stability gates before LP solves."""
    unit = np.asarray(unit, dtype=float)
    mapped = perturbation_map(unit, radius)
    record = {
        "id": str(record_id),
        "radius": float(radius),
        "unit": unit.tolist(),
        "speed_kmh": mapped["speed_kmh"],
        "mass_fraction": mapped["mass_fraction"],
        "payload_height": mapped["payload_height"],
        "physical_overrides": {
            name: float(value) for name, value in mapped["overrides"].items()
        },
        "plant_multipliers": mapped["plant_multipliers"],
        "gain_multipliers": mapped["gain_multipliers"].tolist(),
        "fir_multipliers": mapped["fir_multipliers"].tolist(),
        "finite": False,
        "proposed_stable": False,
        "baseline_stable": False,
        "support_success": False,
        "feasible": False,
        "rear_pass": False,
        "pass": False,
        "error": None,
    }
    try:
        proposed = ParametricStudy(
            speed_kmh=mapped["speed_kmh"], overrides=mapped["overrides"],
            feedback_gain=mapped["gain"], phases=9, label=record_id,
        )
        baseline = ParametricStudy(
            speed_kmh=mapped["speed_kmh"], overrides=mapped["overrides"],
            feedback_gain=mapped["baseline_gain"], phases=9,
            label=f"{record_id}-baseline",
        )
        record["finite"] = bool(finite_study(proposed) and finite_study(baseline))
        record["spectral_radius"] = float(proposed.rho_closed)
        record["baseline_spectral_radius"] = float(baseline.rho_closed)
        record["proposed_stable"] = bool(proposed.rho_closed < 1.0)
        record["baseline_stable"] = bool(baseline.rho_closed < 1.0)
        if not record["finite"] or not record["proposed_stable"]:
            return record

        outputs = evaluate_all_outputs(proposed, mapped["fir"], output_phases)
        record.update(outputs)
        record["feasible"] = bool(outputs["max_utilization"] <= 1.0)
        if record["baseline_stable"]:
            rear = rear_certificate(proposed, baseline, mapped["fir"], rear_phases)
            record["rear_certificate"] = rear
            record["rear_pass"] = bool(rear["relative_margin"] > 0.0)
        record["support_success"] = True
        record["pass"] = bool(
            record["finite"] and record["proposed_stable"] and
            record["baseline_stable"] and record["support_success"] and
            record["feasible"] and record["rear_pass"]
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def summarize_bank(records: list[dict]) -> dict:
    stable = [record for record in records if record["proposed_stable"]]
    support = [record for record in stable if record["support_success"]]
    rear_values = [record for record in support if "rear_certificate" in record]
    summary = {
        "sample_count": len(records),
        "finite_count": sum(record["finite"] for record in records),
        "error_count": sum(record["error"] is not None for record in records),
        "proposed_stable_count": len(stable),
        "baseline_stable_count": sum(record["baseline_stable"] for record in records),
        "support_success_count": len(support),
        "feasible_count": sum(record["feasible"] for record in records),
        "rear_pass_count": sum(record["rear_pass"] for record in records),
        "full_pass_count": sum(record["pass"] for record in records),
        "current_limiting_count": sum(
            record.get("limiting_constraint") in ("front_current", "rear_current")
            for record in stable
        ),
    }
    if records:
        rho_record = max(records, key=lambda r: r.get("spectral_radius", -np.inf))
        summary["max_spectral_radius"] = float(rho_record["spectral_radius"])
        summary["max_spectral_radius_id"] = rho_record["id"]
    if support:
        ltr_record = max(support, key=lambda r: r["worst_ltr"])
        util_record = max(support, key=lambda r: r["max_utilization"])
        summary.update({
            "screen_worst_stable_ltr": float(ltr_record["worst_ltr"]),
            "screen_worst_stable_ltr_id": ltr_record["id"],
            "screen_max_utilization": float(util_record["max_utilization"]),
            "screen_max_utilization_id": util_record["id"],
            "screen_max_utilization_constraint": util_record["limiting_constraint"],
        })
    if rear_values:
        rear_record = min(
            rear_values, key=lambda r: r["rear_certificate"]["relative_margin"]
        )
        summary.update({
            "screen_min_rear_margin": float(
                rear_record["rear_certificate"]["relative_margin"]
            ),
            "screen_min_rear_margin_percent": float(
                rear_record["rear_certificate"]["relative_margin_percent"]
            ),
            "screen_min_rear_margin_id": rear_record["id"],
        })
    return summary