"""Configurable fixed-controller studies for the version-8 sensitivity audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm
from scipy.stats import qmc


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V4_PATH = ROOT / "certified_aarb_v4" / "study.py"
_spec = importlib.util.spec_from_file_location("certified_aarb_v4_sensitivity_base", V4_PATH)
v4 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = v4
_spec.loader.exec_module(v4)

base = v4.base
p = v4.p
distributed_payload_overrides = v4.distributed_payload_overrides
foh_zoh_lift = base.foh_zoh_lift


class ConfigurableStudy(v4.ParametricStudy):
    """Parametric plant with configurable horizon and steering family.

    The physical feedback gain and FIR coefficients remain fixed.  When the
    steering amplitude changes, the FIR input is scaled against the original
    two-degree normalization so the implemented physical steering-to-current
    gain is unchanged.
    """

    def __init__(
        self, *, steering_amp_rad=base.AMP, steering_rate_rad_s=base.RATE,
        delays=base.DELAYS, n=base.N, **kwargs
    ):
        self.steering_amp = float(steering_amp_rad)
        self.steering_rate = float(steering_rate_rad_s)
        self.delays = np.asarray(delays, dtype=int).copy()
        if self.steering_amp <= 0.0 or self.steering_rate <= 0.0:
            raise ValueError("steering amplitude and rate must be positive")
        if self.delays.shape != (6,):
            raise ValueError("the implemented FIR requires six declared delays")
        self.policy_input_scale = self.steering_amp / base.AMP
        super().__init__(n=int(n), **kwargs)

    def _steering_polytope(self):
        rho = self.steering_rate * self.h / self.steering_amp
        rows, rhs = [], []
        for k in range(self.n):
            row = np.zeros(self.n)
            row[k] = 1.0
            if k:
                row[k - 1] = -1.0
            rows.extend([row, -row])
            rhs.extend([rho, rho])
        return np.asarray(rows), np.asarray(rhs)

    def _build_start_maps(self):
        X = np.zeros((13, 10, self.n))
        for k in range(self.n - 1):
            X = np.einsum("ab,qbn->qan", self.ac, X)
            X[0] += np.outer(self.g0 * self.steering_amp, self._selector(k))
            X[0] += np.outer(self.g1 * self.steering_amp, self._selector(k + 1))
            for j in range(12):
                channel = j // 6
                delay = self.delays[j % 6]
                X[j + 1] += np.outer(
                    self.bd_phys[:, channel] * base.I0 * self.policy_input_scale,
                    self._selector(k - delay),
                )
        k = self.n - 1
        U = -np.einsum("ab,qbn->qan", self.K, X)
        for j in range(12):
            channel = j // 6
            delay = self.delays[j % 6]
            U[j + 1, channel] += (
                base.I0 * self.policy_input_scale * self._selector(k - delay)
            )
        return X, U

    def maps(self, phases=None):
        phases = self.phase_times if phases is None else np.asarray(phases)
        key = tuple(np.round(phases, 14))
        if key in self._map_cache:
            return self._map_cache[key]
        X, U = self._start_maps
        all_maps = np.zeros((len(phases), 16, 13, self.n))
        k = self.n - 1
        for qphase, tau in enumerate(phases):
            At, Bt, G0t, G1t = foh_zoh_lift(
                self.A, self.B, self.D, float(tau), self.h
            )
            Xp = np.einsum("ab,qbn->qan", At, X)
            Xp += np.einsum("ab,qbn->qan", Bt, U)
            Xp[0] += np.outer(G0t * self.steering_amp, self._selector(k))
            Xp[0] += np.outer(G1t * self.steering_amp, self._selector(k + 1))
            g = np.einsum("ia,qan->iqn", self.C, Xp)
            g += np.einsum("ia,qan->iqn", self.Du, U)
            lam = float(tau / self.h)
            delta_map = self.steering_amp * (
                (1.0 - lam) * self._selector(k) + lam * self._selector(k + 1)
            )
            g[:, 0] += self.Ddelta[:, None] * delta_map[None, :]
            all_maps[qphase] = g
        self._map_cache[key] = all_maps
        return all_maps

    def _intersample_correction(self, f, spacing):
        X, U = self._start_maps
        Xc = X[0] + np.einsum("q,qan->an", f, X[1:])
        Uc = U[0] + np.einsum("q,qan->an", f, U[1:])
        xbar = np.asarray([self.support(row) for row in Xc]) / base.STATE_SCALE
        ubar = np.asarray([self.support(row) for row in Uc]) / base.I0
        vbar = np.r_[xbar, ubar, self.steering_amp, self.steering_rate]
        SinvAS = np.diag(1.0 / base.STATE_SCALE) @ self.A @ np.diag(base.STATE_SCALE)
        SinvB = np.diag(1.0 / base.STATE_SCALE) @ self.B * base.I0
        SinvD = np.diag(1.0 / base.STATE_SCALE) @ self.D[:, 0]
        J = np.zeros((14, 14))
        J[:10, :10] = SinvAS
        J[:10, 10:12] = SinvB
        J[:10, 12] = SinvD
        J[12, 13] = 1.0
        Caug = np.zeros((16, 14))
        Caug[:, :10] = self.C @ np.diag(base.STATE_SCALE)
        Caug[:, 10:12] = self.Du * base.I0
        Caug[:, 12] = self.Ddelta
        envelope = expm(np.abs(J) * self.h) @ vbar
        second = np.abs(Caug @ (J @ J)) @ envelope
        return second * spacing * spacing / 8.0, second, vbar


def all_operating_points():
    structured = []
    for speed in np.linspace(50.0, 75.0, 6):
        structured.append({
            "set": "structured", "speed_kmh": float(speed),
            "mass_fraction": 0.0, "payload_height": float(p.h),
        })
        for fraction in (0.05, 0.10):
            for height in (p.h, 1.475, 1.80):
                structured.append({
                    "set": "structured", "speed_kmh": float(speed),
                    "mass_fraction": float(fraction),
                    "payload_height": float(height),
                })
    sampler = qmc.Sobol(d=3, scramble=True, seed=20260919)
    holdout = []
    for row in sampler.random_base2(m=7):
        holdout.append({
            "set": "sobol_holdout",
            "speed_kmh": float(50.0 + 25.0 * row[0]),
            "mass_fraction": float(0.10 * row[1]),
            "payload_height": float(p.h + (1.80 - p.h) * row[2]),
        })
    return structured + holdout


def structured_points():
    return all_operating_points()[:42]


def skeleton_points():
    rows = []
    for speed in (50.0, 62.5, 75.0):
        for fraction, height, label in (
            (0.0, p.h, "nominal"),
            (0.10, p.h, "payload-low"),
            (0.10, 1.80, "payload-high"),
        ):
            rows.append({
                "set": "design_skeleton", "label": f"{speed:g} km/h; {label}",
                "speed_kmh": speed, "mass_fraction": fraction,
                "payload_height": height,
            })
    return rows


def point_key(point):
    return (
        point["set"], round(float(point["speed_kmh"]), 12),
        round(float(point["mass_fraction"]), 12),
        round(float(point["payload_height"]), 12),
    )


def build_study(point, feedback_gain, *, n=base.N, amp=base.AMP,
                rate=base.RATE, extra_overrides=None):
    overrides = distributed_payload_overrides(
        point["mass_fraction"], point["payload_height"]
    )
    overrides.update(dict(extra_overrides or {}))
    return ConfigurableStudy(
        speed_kmh=point["speed_kmh"], overrides=overrides,
        feedback_gain=np.asarray(feedback_gain), phases=9, n=n,
        steering_amp_rad=amp, steering_rate_rad_s=rate,
        label=point.get("label"),
    )


def rear_margin_screen(study, coefficients, phases=41):
    phase_times = np.linspace(0.0, study.h, int(phases))
    maps = study.maps(phase_times)
    baseline_g = study.combine_maps(maps, np.zeros(12))
    proposed_g = study.combine_maps(maps, np.asarray(coefficients))
    baseline_feasible = []
    proposed_corrected = []
    for phase in range(int(phases)):
        _, _, primal, _ = study.support(
            baseline_g[phase, 1], return_witness=True
        )
        upper, _, _, _ = study.support(
            proposed_g[phase, 1], return_witness=True
        )
        baseline_feasible.append(float(primal))
        proposed_corrected.append(float(upper))
    interpolation, _, _ = study._intersample_correction(
        np.asarray(coefficients), study.h / (int(phases) - 1)
    )
    baseline = max(baseline_feasible)
    proposed = max(proposed_corrected) + float(interpolation[1])
    margin = baseline - proposed
    return {
        "baseline_feasible_lower": baseline,
        "proposed_continuous_upper": proposed,
        "certified_margin": margin,
        "relative_margin": margin / baseline,
        "relative_margin_percent": 100.0 * margin / baseline,
    }


def evaluate_point(study, coefficients, broad_phases=21, rear_phases=41):
    record = {"spectral_radius": float(study.rho_closed)}
    if study.rho_closed >= 1.0:
        record.update({"status": "FAIL_UNSTABLE", "continuous": None})
        return record
    evaluation = study.evaluate(np.asarray(coefficients), phases=broad_phases)
    continuous = dict(zip(study.names, map(float, evaluation["continuous_upper"])))
    non_ltr = {name: value for name, value in continuous.items()
               if name not in ("front_LTR", "rear_LTR")}
    limiting = max(non_ltr, key=non_ltr.get)
    record.update({
        "status": "PASS",
        "continuous": continuous,
        "front_LTR": continuous["front_LTR"],
        "rear_LTR": continuous["rear_LTR"],
        "worst_LTR": max(continuous["front_LTR"], continuous["rear_LTR"]),
        "max_constraint_utilization": non_ltr[limiting],
        "limiting_constraint": limiting,
        "rear_margin": rear_margin_screen(study, coefficients, rear_phases),
        "max_support_correction": float(np.max(evaluation["support_corrections"])),
    })
    return record