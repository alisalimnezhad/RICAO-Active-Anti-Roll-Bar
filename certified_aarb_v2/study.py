"""Reconstructed finite-horizon AARB certificate study.

The ten-state plant and nominal LQR gain are imported unchanged from validation/.
All new computations are isolated in this directory.  Steering is FOH, current
is ZOH, and the controller uses only current/past steering samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
if str(VALIDATION) not in sys.path:
    sys.path.insert(0, str(VALIDATION))

import parameters as p  # noqa: E402
from vehicle_model import build_model  # noqa: E402
from lqr_controller import design_lqr, STATE_SCALE  # noqa: E402

H = 0.02
N = 200
AMP = np.deg2rad(2.0)
RATE = np.deg2rad(5.0)
RHO = RATE * H / AMP
I0 = 0.02
DELAYS = np.array([0, 2, 5, 10, 20, 40], dtype=int)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(value), indent=2, sort_keys=True))


def foh_zoh_lift(A, B, D, tau, hold=H):
    """Exact x(tau)=Ad*x0+Bd*u+G0*d0+G1*d1 for a hold of length hold."""
    n, m = B.shape
    M = np.zeros((n + m + 2, n + m + 2))
    M[:n, :n] = A
    M[:n, n:n + m] = B
    M[:n, n + m] = D[:, 0]
    M[n + m, n + m + 1] = 1.0
    E = expm(M * tau)
    Ad = E[:n, :n]
    Bd = E[:n, n:n + m]
    Ed = E[:n, n + m]
    Es = E[:n, n + m + 1]
    G1 = Es / hold
    G0 = Ed - G1
    return Ad, Bd, G0, G1


@dataclass
class MasterResult:
    f: np.ndarray
    gamma: float
    lower: float
    primal_dual_gap: float
    max_inequality_residual: float


class Study:
    def __init__(self, phases=9, n=N):
        self.h = H
        self.n = n
        self.phases = phases
        self.phase_times = np.linspace(0.0, H, phases)
        self.model = build_model()
        self.A = self.model["A"]
        self.B = self.model["B"]
        self.D = self.model["D"]
        self.K, self.Q, self.R, self.care = design_lqr(self.model)
        self.ad, self.bd_phys, self.g0, self.g1 = foh_zoh_lift(
            self.A, self.B, self.D, H
        )
        self.bd = self.bd_phys * I0
        self.ac = self.ad - self.bd_phys @ self.K
        self.rho_closed = float(np.max(np.abs(np.linalg.eigvals(self.ac))))
        self.C, self.Du, self.Ddelta, self.names = self._output_definition()
        self.Aw, self.bw = self._steering_polytope()
        self._start_maps = self._build_start_maps()
        self._map_cache = {}

    def _output_definition(self):
        C = np.zeros((16, 10))
        Du = np.zeros((16, 2))
        Dd = np.zeros(16)
        C[:2] = self.model["C"]
        Du[2, 0] = 1.0 / I0
        Du[3, 1] = 1.0 / I0
        C[4, 6] = p.area / 120000.0
        C[5, 8] = p.area / 120000.0
        C[6, 7] = 1.0 / 0.000485
        C[7, 9] = 1.0 / 0.000485
        C[8, 6] = -p.KP / 0.0022
        C[8, 7] = p.Kx / 0.0022
        C[9, 8] = -p.KP / 0.0022
        C[9, 9] = p.Kx / 0.0022
        C[10, [2, 4]] = [1.0, -1.0]
        C[11, [2, 5]] = [1.0, -1.0]
        C[10:12] /= np.deg2rad(7.0)
        C[12, 0] = 1.0 / np.deg2rad(3.0)
        C[13, 2] = 1.0 / np.deg2rad(5.0)
        C[14, 0] = -1.0 / np.deg2rad(3.0)
        C[14, 1] = -p.lf / p.v / np.deg2rad(3.0)
        Dd[14] = 1.0 / np.deg2rad(3.0)
        C[15, 0] = -1.0 / np.deg2rad(3.0)
        C[15, 1] = p.lr / p.v / np.deg2rad(3.0)
        names = [
            "front_LTR", "rear_LTR", "front_current", "rear_current",
            "front_force", "rear_force", "front_spool", "rear_spool",
            "front_flow", "rear_flow", "front_suspension_roll",
            "rear_suspension_roll", "sideslip", "body_roll",
            "front_tyre_slip", "rear_tyre_slip",
        ]
        return C, Du, Dd, names

    def _steering_polytope(self):
        rows, rhs = [], []
        for k in range(self.n):
            r = np.zeros(self.n)
            r[k] = 1.0
            if k:
                r[k - 1] = -1.0
            rows.extend([r, -r])
            rhs.extend([RHO, RHO])
        return np.array(rows), np.array(rhs)

    def support(self, g, return_witness=False):
        result = linprog(
            -np.asarray(g), A_ub=self.Aw, b_ub=self.bw,
            bounds=[(-1.0, 1.0)] * self.n, method="highs"
        )
        if not result.success:
            raise RuntimeError(result.message)
        upper = float(-result.fun)
        y = np.minimum(result.ineqlin.marginals, 0.0)
        residual = -np.asarray(g) - self.Aw.T @ y
        # min(-g'w) >= b'y + sum_j min(r_j*l_j,r_j*u_j)
        lower_min = float(self.bw @ y - np.abs(residual).sum())
        corrected_upper = -lower_min
        payload = (corrected_upper, result.x, upper, float(corrected_upper - upper))
        return payload if return_witness else corrected_upper

    def _selector(self, knot):
        row = np.zeros(self.n)
        if 1 <= knot <= self.n:
            row[knot - 1] = 1.0
        return row

    def _build_start_maps(self):
        """Affine maps at t_(N-1): basis 0 is plant steering; 1:13 are FIR."""
        X = np.zeros((13, 10, self.n))
        for k in range(self.n - 1):
            X = np.einsum("ab,qbn->qan", self.ac, X)
            X[0] += np.outer(self.g0 * AMP, self._selector(k))
            X[0] += np.outer(self.g1 * AMP, self._selector(k + 1))
            for j in range(12):
                channel = j // 6
                delay = DELAYS[j % 6]
                X[j + 1] += np.outer(
                    self.bd_phys[:, channel] * I0,
                    self._selector(k - delay),
                )
        k = self.n - 1
        U = -np.einsum("ab,qbn->qan", self.K, X)
        for j in range(12):
            channel = j // 6
            delay = DELAYS[j % 6]
            U[j + 1, channel] += I0 * self._selector(k - delay)
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
            Xp[0] += np.outer(G0t * AMP, self._selector(k))
            Xp[0] += np.outer(G1t * AMP, self._selector(k + 1))
            g = np.einsum("ia,qan->iqn", self.C, Xp)
            g += np.einsum("ia,qan->iqn", self.Du, U)
            lam = float(tau / self.h)
            delta_map = AMP * ((1.0 - lam) * self._selector(k) + lam * self._selector(k + 1))
            g[:, 0] += self.Ddelta[:, None] * delta_map[None, :]
            all_maps[qphase] = g
        self._map_cache[key] = all_maps
        return all_maps

    @staticmethod
    def combine_maps(maps, f):
        return maps[:, :, 0] + np.einsum("piqn,q->pin", maps[:, :, 1:], f)

    def _master_arrays(self, witnesses, margin=0.998):
        maps = self.maps()
        rows, rhs = [], []
        for w in witnesses:
            vals = np.einsum("piqn,n->piq", maps, w)
            for phase in range(self.phases):
                for output in range(16):
                    for sign in (-1.0, 1.0):
                        row = np.zeros(13)
                        row[:12] = sign * vals[phase, output, 1:]
                        base = sign * vals[phase, output, 0]
                        if output < 2:
                            row[12] = -1.0
                            bound = -base
                        else:
                            bound = margin - base
                        rows.append(row)
                        rhs.append(bound)
        return np.asarray(rows), np.asarray(rhs)

    def solve_master(self, witnesses, margin=0.998):
        A, b = self._master_arrays(witnesses, margin)
        c = np.r_[np.zeros(12), 1.0]
        lo = np.r_[np.full(12, -50.0), 0.0]
        hi = np.r_[np.full(12, 50.0), 5.0]
        result = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lo, hi)), method="highs")
        if not result.success:
            raise RuntimeError(result.message)
        y = np.minimum(result.ineqlin.marginals, 0.0)
        residual = c - A.T @ y
        lower = float(b @ y + np.where(residual >= 0.0, residual * lo, residual * hi).sum())
        return MasterResult(
            f=result.x[:12], gamma=float(result.x[12]), lower=lower,
            primal_dual_gap=float(result.fun - lower),
            max_inequality_residual=float(np.max(A @ result.x - b)),
        )

    def initial_witnesses(self):
        k = np.arange(1, self.n + 1)
        ramp = np.minimum(k * RHO, 1.0)
        sine = 0.30 * np.sin(2.0 * np.pi * k * self.h / 2.0)
        assert np.max(np.abs(np.diff(np.r_[0.0, sine]))) <= RHO + 1e-12
        return [ramp, sine]

    def synthesize(self, tolerance=2e-6, max_iterations=220, margin=0.998, verbose=False):
        witnesses = self.initial_witnesses()
        history = []
        maps = self.maps()
        for iteration in range(1, max_iterations + 1):
            master = self.solve_master(witnesses, margin=margin)
            g = self.combine_maps(maps, master.f)
            worst = None
            maxima = np.zeros((self.phases, 16))
            for phase in range(self.phases):
                for output in range(16):
                    upper, w, primal, correction = self.support(
                        g[phase, output], return_witness=True
                    )
                    maxima[phase, output] = upper
                    cap = master.gamma if output < 2 else margin
                    violation = upper - cap
                    if worst is None or violation > worst[0]:
                        worst = (violation, w, phase, output, primal, correction)
            history.append({
                "iteration": iteration,
                "master_gamma": master.gamma,
                "master_lower": master.lower,
                "max_separation_violation": worst[0],
                "worst_phase": worst[2],
                "worst_output": self.names[worst[3]],
                "witness_count": len(witnesses),
                "max_phase_LTR": float(np.max(maxima[:, :2])),
                "max_phase_constraint": float(np.max(maxima[:, 2:])),
            })
            if verbose and (iteration == 1 or iteration % 10 == 0 or worst[0] <= tolerance):
                print("exchange", iteration, "gamma", master.gamma,
                      "violation", worst[0], "output", self.names[worst[3]], flush=True)
            if worst[0] <= tolerance:
                break
            witnesses.append(worst[1])
        else:
            raise RuntimeError("exchange algorithm did not converge")
        family = self.solve_master(witnesses, margin=1.0)
        return master.f, witnesses, history, master, family

    def _intersample_correction(self, f, spacing):
        X, U = self._start_maps
        Xc = X[0] + np.einsum("q, qan->an", f, X[1:])
        Uc = U[0] + np.einsum("q, qan->an", f, U[1:])
        xbar = np.array([self.support(row) for row in Xc]) / STATE_SCALE
        ubar = np.array([self.support(row) for row in Uc]) / I0
        vbar = np.r_[xbar, ubar, AMP, RATE]
        SinvAS = np.diag(1.0 / STATE_SCALE) @ self.A @ np.diag(STATE_SCALE)
        SinvB = np.diag(1.0 / STATE_SCALE) @ self.B * I0
        SinvD = np.diag(1.0 / STATE_SCALE) @ self.D[:, 0]
        J = np.zeros((14, 14))
        J[:10, :10] = SinvAS
        J[:10, 10:12] = SinvB
        J[:10, 12] = SinvD
        J[12, 13] = 1.0
        Caug = np.zeros((16, 14))
        Caug[:, :10] = self.C @ np.diag(STATE_SCALE)
        Caug[:, 10:12] = self.Du * I0
        Caug[:, 12] = self.Ddelta
        envelope = expm(np.abs(J) * self.h) @ vbar
        second = np.abs(Caug @ (J @ J)) @ envelope
        return second * spacing * spacing / 8.0, second, vbar

    def evaluate(self, f, phases=41):
        phase_times = np.linspace(0.0, self.h, phases)
        maps = self.maps(phase_times)
        g = self.combine_maps(maps, np.asarray(f))
        upper = np.zeros((phases, 16))
        primal = np.zeros_like(upper)
        corrections = np.zeros_like(upper)
        witnesses = [[None] * 16 for _ in range(phases)]
        for phase in range(phases):
            for output in range(16):
                u, w, pr, corr = self.support(g[phase, output], return_witness=True)
                upper[phase, output] = u
                primal[phase, output] = pr
                corrections[phase, output] = corr
                witnesses[phase][output] = w
        spacing = self.h / (phases - 1)
        interpolation, second, vbar = self._intersample_correction(f, spacing)
        continuous = np.max(upper, axis=0) + interpolation
        return {
            "phase_upper": upper,
            "phase_primal": primal,
            "support_corrections": corrections,
            "grid_upper": np.max(upper, axis=0),
            "interpolation_correction": interpolation,
            "continuous_upper": continuous,
            "second_derivative_bound": second,
            "interval_start_bound": vbar,
            "witnesses": witnesses,
            "phase_times": phase_times,
        }

    def simulate(self, f, w, refine=20):
        f = np.asarray(f)
        wfull = np.r_[0.0, np.asarray(w)]
        x = np.zeros(10)
        times, states, currents, steering = [], [], [], []
        for k in range(self.n):
            feed = np.zeros(2)
            for j, delay in enumerate(DELAYS):
                idx = k - delay
                if idx >= 0:
                    feed += f.reshape(2, 6)[:, j] * wfull[idx]
            u = -self.K @ x + I0 * feed
            for q in range(refine):
                tau = q * self.h / refine
                At, Bt, G0t, G1t = foh_zoh_lift(self.A, self.B, self.D, tau, self.h)
                xp = At @ x + Bt @ u + G0t * (AMP * wfull[k]) + G1t * (AMP * wfull[k + 1])
                delta = AMP * ((1.0 - tau / self.h) * wfull[k] + (tau / self.h) * wfull[k + 1])
                times.append(k * self.h + tau)
                states.append(xp)
                currents.append(u)
                steering.append(delta)
            x = self.ad @ x + self.bd_phys @ u + self.g0 * (AMP * wfull[k]) + self.g1 * (AMP * wfull[k + 1])
        return {
            "time": np.asarray(times), "state": np.asarray(states),
            "current": np.asarray(currents), "steering": np.asarray(steering),
        }

    def output_history(self, trajectory):
        x = trajectory["state"]
        u = trajectory["current"]
        d = trajectory["steering"]
        return x @ self.C.T + u @ self.Du.T + d[:, None] * self.Ddelta[None, :]


def causal_tree_lower(study: Study, fraction=1.0, segments=6, width=5):
    """Relaxed, nonanticipative 0.6 s tree lower bound from the manuscript."""
    from scipy import sparse

    nodes = [dict(x=np.zeros(10), G=np.zeros((10, 0)), ids=[], delta=0.0, slope=0.0)]
    constraints = []
    ncontrol = 0
    count = 1
    for k in range(segments * width):
        next_nodes = []
        for node in nodes:
            ids = node["ids"] + [ncontrol, ncontrol + 1]
            ncontrol += 2
            gx = np.column_stack([study.ad @ node["G"], study.bd])
            xx = study.ad @ node["x"]
            if k % width == 0:
                slopes = [r for r in (-RATE, RATE)
                          if abs(node["delta"] + r * width * H) <= AMP + 1e-12]
            else:
                slopes = [node["slope"]]
            for slope in slopes:
                delta = node["delta"] + slope * H
                x = xx + study.g0 * node["delta"] + study.g1 * delta
                for ci in study.C[:2]:
                    constraints.append((ids, ci @ gx, float(ci @ x)))
                next_nodes.append(dict(x=x, G=gx, ids=ids, delta=delta, slope=slope))
                count += 1
        nodes = next_nodes
    row, col, data, b = [], [], [], []
    for ids, vals, offset in constraints:
        for sign in (-1.0, 1.0):
            r = len(b)
            row.extend([r] * (len(ids) + 1))
            col.extend(ids + [ncontrol])
            data.extend(list(sign * vals) + [-1.0])
            b.append(-sign * offset)
    A = sparse.csr_matrix((data, (row, col)), shape=(len(b), ncontrol + 1))
    b = np.asarray(b)
    c = np.r_[np.zeros(ncontrol), 1.0]
    lo = np.r_[np.full(ncontrol, -fraction), 0.0]
    hi = np.r_[np.full(ncontrol, fraction), 2.0]
    result = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lo, hi)), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    y = np.minimum(result.ineqlin.marginals, 0.0)
    residual = c - A.T @ y
    lower = float(b @ y + np.where(residual >= 0.0, residual * lo, residual * hi).sum())
    return {
        "lower": lower, "primal": float(result.fun),
        "duality_gap": float(result.fun - lower), "node_count": count,
        "leaf_count": len(nodes), "variables": ncontrol + 1,
        "duration": segments * width * H,
    }