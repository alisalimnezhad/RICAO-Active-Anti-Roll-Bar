"""Stage-3 certified design utilities.

This module extends the independently reconstructed version-2 calculation
without changing its validated ten-state plant source.  It adds

* axle-specific robust objectives and caps for a certified Pareto study;
* explicit peak-current authority caps for fair controller comparisons; and
* joint synthesis over a finite, declared bank of speed/load models.

The finite model bank is an engineering stress-test set.  It is not a proof
over a continuous parameter box.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "certified_aarb_v2" / "study.py"
_spec = importlib.util.spec_from_file_location("certified_aarb_v2_study", V2)
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

BaseStudy = base.Study
MasterResult = base.MasterResult
H, N, AMP, RATE, I0, DELAYS = base.H, base.N, base.AMP, base.RATE, base.I0, base.DELAYS
p = base.p


@contextmanager
def temporary_parameters(overrides):
    """Temporarily override the shared validation parameter module."""
    saved = {name: getattr(p, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(p, name, float(value))
        yield
    finally:
        for name, value in saved.items():
            setattr(p, name, value)


class ParametricStudy(BaseStudy):
    """A version-2 Study constructed at declared parameter values.

    If ``feedback_gain`` is supplied, that physical SI gain is retained while
    the plant changes.  This is the fixed-controller case.  Otherwise the
    reference LQR1 gain is redesigned for the declared plant.
    """

    def __init__(self, *, speed_kmh=70.0, overrides=None, feedback_gain=None,
                 phases=9, n=N, label=None):
        overrides = dict(overrides or {})
        overrides["v"] = float(speed_kmh) / 3.6
        with temporary_parameters(overrides):
            super().__init__(phases=phases, n=n)
        if feedback_gain is not None:
            self.K = np.asarray(feedback_gain, dtype=float).copy()
            self.ac = self.ad - self.bd_phys @ self.K
            self.rho_closed = float(np.max(np.abs(np.linalg.eigvals(self.ac))))
            self._start_maps = self._build_start_maps()
            self._map_cache = {}
        self.speed_kmh = float(speed_kmh)
        self.parameter_overrides = overrides
        self.label = label or f"{speed_kmh:g} km/h"

    def support(self, g, return_witness=False):
        """Robust support solve with deterministic HiGHS fallbacks.

        Very occasionally the automatic HiGHS dispatcher returns an
        ``Unknown`` model status despite a feasible primal point.  A fresh
        dual-simplex or interior-point solve removes that implementation
        artifact; only a formally successful solve is used for a certificate.
        """
        vector = np.asarray(g, dtype=float)
        attempts = (
            ("highs", None),
            ("highs-ds", {"presolve": False}),
            ("highs-ipm", {"presolve": True}),
        )
        messages = []
        for method, options in attempts:
            result = linprog(
                -vector, A_ub=self.Aw, b_ub=self.bw,
                bounds=[(-1.0, 1.0)] * self.n, method=method,
                options=options,
            )
            if not result.success:
                messages.append(f"{method}: {result.message}")
                continue
            upper = float(-result.fun)
            y = np.minimum(result.ineqlin.marginals, 0.0)
            residual = -vector - self.Aw.T @ y
            lower_min = float(self.bw @ y - np.abs(residual).sum())
            corrected_upper = -lower_min
            payload = (corrected_upper, result.x, upper,
                       float(corrected_upper - upper))
            return payload if return_witness else corrected_upper
        raise RuntimeError("; ".join(messages))


def central_payload_overrides(mass_fraction, payload_height):
    """Return a mechanically coupled central point-payload idealization.

    The added mass lies on the vehicle longitudinal centreline and at the
    nominal longitudinal CG, so lf/lr and yaw inertia are unchanged.  Sprung
    mass, total mass, CG height and roll inertia are updated together using
    the parallel-axis theorem.  The idealization is intentionally narrow and
    is not presented as an identified loading condition.
    """
    mp = float(mass_fraction) * p.ms
    ms_new = p.ms + mp
    h_new = (p.ms * p.h + mp * float(payload_height)) / ms_new
    ixx_new = p.Ixx + p.ms * (p.h - h_new) ** 2 + mp * (float(payload_height) - h_new) ** 2
    return {
        "ms": ms_new,
        "m": ms_new + p.muf + p.mur,
        "h": h_new,
        "Ixx": ixx_new,
    }


def proportional_mass_overrides(mass_fraction):
    """Uniform sprung-body mass scaling with fixed geometry and inertia shape."""
    scale = 1.0 + float(mass_fraction)
    ms_new = p.ms * scale
    return {
        "ms": ms_new,
        "m": ms_new + p.muf + p.mur,
        "Ixx": p.Ixx * scale,
        "Ixz": p.Ixz * scale,
        "Izz": p.Izz * scale,
    }


@dataclass
class RobustResult:
    f: np.ndarray
    gamma: float
    lower: float
    scenarios: list
    history: list
    master: MasterResult


class RobustDesigner:
    """Exchange LP for one or more finite plant models."""

    def __init__(self, studies, objective_outputs=(0, 1), caps=None,
                 coefficient_bound=50.0):
        self.studies = list(studies)
        if not self.studies:
            raise ValueError("at least one study is required")
        self.objective_outputs = tuple(int(i) for i in objective_outputs)
        if any(i < 0 or i >= 16 for i in self.objective_outputs):
            raise ValueError("invalid objective output")
        if caps is None:
            caps = np.ones((len(self.studies), 16)) * 0.998
        self.caps = np.broadcast_to(np.asarray(caps, dtype=float),
                                    (len(self.studies), 16)).copy()
        self.coefficient_bound = float(coefficient_bound)

    def _master_arrays(self, scenarios):
        rows, rhs = [], []
        for model_index, witness in scenarios:
            study = self.studies[model_index]
            values = np.einsum("piqn,n->piq", study.maps(), witness)
            for phase in range(study.phases):
                for output in range(16):
                    for sign in (-1.0, 1.0):
                        row = np.zeros(13)
                        row[:12] = sign * values[phase, output, 1:]
                        base_value = sign * values[phase, output, 0]
                        if output in self.objective_outputs:
                            row[12] = -1.0
                            bound = -base_value
                        else:
                            bound = self.caps[model_index, output] - base_value
                        rows.append(row)
                        rhs.append(bound)
        return np.asarray(rows), np.asarray(rhs)

    def solve_master(self, scenarios):
        A, b = self._master_arrays(scenarios)
        c = np.r_[np.zeros(12), 1.0]
        lo = np.r_[np.full(12, -self.coefficient_bound), 0.0]
        hi = np.r_[np.full(12, self.coefficient_bound), 5.0]
        result = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lo, hi)), method="highs")
        if not result.success:
            raise RuntimeError(result.message)
        y = np.minimum(result.ineqlin.marginals, 0.0)
        residual = c - A.T @ y
        lower = float(b @ y + np.where(residual >= 0.0,
                                       residual * lo, residual * hi).sum())
        return MasterResult(
            f=result.x[:12], gamma=float(result.x[12]), lower=lower,
            primal_dual_gap=float(result.fun - lower),
            max_inequality_residual=float(np.max(A @ result.x - b)),
        )

    @staticmethod
    def _scenario_key(model_index, witness):
        return model_index, np.round(np.asarray(witness), 11).tobytes()

    def seed_scenarios(self, candidate_coefficients):
        """Collect one active support witness per output/model/candidate."""
        scenarios, keys = [], set()
        for model_index, study in enumerate(self.studies):
            maps = study.maps()
            for f in candidate_coefficients:
                g = study.combine_maps(maps, np.asarray(f))
                for output in range(16):
                    best = None
                    for phase in range(study.phases):
                        upper, witness, _, _ = study.support(
                            g[phase, output], return_witness=True
                        )
                        if best is None or upper > best[0]:
                            best = (upper, witness)
                    key = self._scenario_key(model_index, best[1])
                    if key not in keys:
                        keys.add(key)
                        scenarios.append((model_index, best[1]))
        return scenarios, keys

    def synthesize(self, candidate_coefficients, tolerance=3e-6,
                   max_iterations=180, verbose=False):
        scenarios, keys = self.seed_scenarios(candidate_coefficients)
        history = []
        for iteration in range(1, max_iterations + 1):
            master = self.solve_master(scenarios)
            worst = None
            model_maxima = []
            for model_index, study in enumerate(self.studies):
                g = study.combine_maps(study.maps(), master.f)
                maxima = np.zeros((study.phases, 16))
                for phase in range(study.phases):
                    for output in range(16):
                        upper, witness, primal, correction = study.support(
                            g[phase, output], return_witness=True
                        )
                        maxima[phase, output] = upper
                        cap = master.gamma if output in self.objective_outputs \
                            else self.caps[model_index, output]
                        violation = upper - cap
                        if worst is None or violation > worst[0]:
                            worst = (violation, model_index, witness, phase,
                                     output, primal, correction)
                model_maxima.append(maxima)
            history.append({
                "iteration": iteration,
                "master_gamma": master.gamma,
                "master_lower": master.lower,
                "max_separation_violation": float(worst[0]),
                "worst_model": self.studies[worst[1]].label,
                "worst_phase": int(worst[3]),
                "worst_output": self.studies[worst[1]].names[worst[4]],
                "scenario_count": len(scenarios),
                "model_maxima": [m.max(axis=0).tolist() for m in model_maxima],
            })
            if verbose and (iteration == 1 or iteration % 10 == 0 or worst[0] <= tolerance):
                print("exchange", iteration, "gamma", master.gamma,
                      "violation", worst[0], "model", self.studies[worst[1]].label,
                      "output", self.studies[worst[1]].names[worst[4]], flush=True)
            if worst[0] <= tolerance:
                break
            key = self._scenario_key(worst[1], worst[2])
            if key in keys:
                # A repeated witness with a residual above tolerance signals
                # numerical stagnation.  Accept only a near-tolerance repeat.
                if worst[0] <= 5.0 * tolerance:
                    break
                raise RuntimeError("exchange stagnated on a repeated witness")
            keys.add(key)
            scenarios.append((worst[1], worst[2]))
        else:
            raise RuntimeError("exchange algorithm did not converge")
        final_master = self.solve_master(scenarios)
        return RobustResult(final_master.f, final_master.gamma,
                            final_master.lower, scenarios, history, final_master)


def base_caps(studies, *, current_caps=None, other_margin=0.994):
    """Create robust design caps with room for the intersample correction."""
    caps = np.ones((len(studies), 16), dtype=float) * float(other_margin)
    caps[:, :2] = np.inf
    if current_caps is not None:
        cc = np.asarray(current_caps, dtype=float)
        if cc.ndim == 0:
            cc = np.array([cc, cc])
        caps[:, 2:4] = cc
    return caps


def evaluate_bank(studies, f, phases=41):
    results = []
    for study in studies:
        evaluation = study.evaluate(np.asarray(f), phases=phases)
        results.append({
            "label": study.label,
            "speed_kmh": study.speed_kmh,
            "spectral_radius": study.rho_closed,
            "continuous": evaluation["continuous_upper"],
            "grid": evaluation["grid_upper"],
            "interpolation": evaluation["interpolation_correction"],
            "evaluation": evaluation,
        })
    return results