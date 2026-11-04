"""Exact lifted model for uniformly delayed state and steering measurements.

At the control update ``t = k h``, the implemented current is

    u_k = -K x(k h - tau) + I0 * F * w_history(k h - tau),

and is held over the next 20 ms interval.  Steering remains first-order held.
For ``0 <= tau <= h``, the delayed state lies in the preceding hold interval,
so it can be lifted exactly without approximating the continuous plant.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V4 = ROOT / "certified_aarb_v4"
if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))

from study import ParametricStudy, base  # noqa: E402


AMP = base.AMP
I0 = base.I0
DELAYS = base.DELAYS
foh_zoh_lift = base.foh_zoh_lift


class DelayedParametricStudy(ParametricStudy):
    """Parametric study with a common sensor/computation latency.

    The latency is applied to the full-state feedback and every steering FIR
    tap.  The plant steering input itself is not delayed: it is the physical
    first-order-held steering trajectory admitted by the declared polytope.
    """

    def __init__(self, *, delay_s=0.0, **kwargs):
        self.delay_s = float(delay_s)
        if self.delay_s < -1e-15 or self.delay_s > base.H + 1e-15:
            raise ValueError("delay must lie in [0, 20] ms")
        self.delay_s = min(max(self.delay_s, 0.0), base.H)
        super().__init__(**kwargs)
        self._set_delayed_spectral_radius()

    @property
    def delay_fraction(self):
        return self.delay_s / self.h

    def _delayed_selector(self, knot):
        """FOH steering value at ``knot*h - delay`` in normalized units."""
        eta = self.delay_fraction
        return eta * self._selector(knot - 1) + (1.0 - eta) * self._selector(knot)

    def _partial_hold(self):
        elapsed = self.h - self.delay_s
        return foh_zoh_lift(self.A, self.B, self.D, elapsed, self.h)

    def _build_start_maps(self):
        """Maps at the final interval start for the delayed controller."""
        # X maps x(kh); Xd maps x(kh-delay).  Basis 0 is physical steering and
        # bases 1:13 are the twelve FIR coefficients.
        X = np.zeros((13, 10, self.n))
        Xd = np.zeros_like(X)
        Aa, Ba, G0a, G1a = self._partial_hold()

        for k in range(self.n):
            U = -np.einsum("ab,qbn->qan", self.K, Xd)
            for j in range(12):
                channel = j // 6
                tap_delay = DELAYS[j % 6]
                U[j + 1, channel] += I0 * self._delayed_selector(k - tap_delay)

            if k == self.n - 1:
                return X, U

            Xnext = np.einsum("ab,qbn->qan", self.ad, X)
            Xnext += np.einsum("ab,qbn->qan", self.bd_phys, U)
            Xnext[0] += np.outer(self.g0 * AMP, self._selector(k))
            Xnext[0] += np.outer(self.g1 * AMP, self._selector(k + 1))

            # At the next update, the delayed state lies h-delay seconds into
            # the current plant interval, under the same ZOH current.
            Xdnext = np.einsum("ab,qbn->qan", Aa, X)
            Xdnext += np.einsum("ab,qbn->qan", Ba, U)
            Xdnext[0] += np.outer(G0a * AMP, self._selector(k))
            Xdnext[0] += np.outer(G1a * AMP, self._selector(k + 1))
            X, Xd = Xnext, Xdnext

        raise RuntimeError("unreachable delayed-map construction state")

    def _set_delayed_spectral_radius(self):
        """Schur radius of the autonomous delayed sampled-data loop."""
        Aa, Ba, _, _ = self._partial_hold()
        augmented = np.block([
            [self.ad, -self.bd_phys @ self.K],
            [Aa, -Ba @ self.K],
        ])
        self.delayed_closed_loop_matrix = augmented
        self.rho_closed = float(np.max(np.abs(np.linalg.eigvals(augmented))))

    def support(self, g, return_witness=False):
        """Scale large unstable-loop objectives before the support LP solve."""
        vector = np.asarray(g, dtype=float)
        scale = max(1.0, float(np.max(np.abs(vector))))
        result = super().support(vector / scale, return_witness=return_witness)
        if not return_witness:
            return float(result) * scale
        upper, witness, primal, correction = result
        return (
            float(upper) * scale,
            witness,
            float(primal) * scale,
            float(correction) * scale,
        )

    def direct_final_interval(self, coefficients, steering, phases):
        """Direct recurrence used only to validate the lifted delayed maps."""
        f = np.asarray(coefficients, dtype=float).reshape(2, 6)
        w = np.asarray(steering, dtype=float)
        if w.shape != (self.n,):
            raise ValueError("steering vector has the wrong length")
        wfull = np.r_[0.0, w]
        x = np.zeros(10)
        xd = np.zeros(10)
        Aa, Ba, G0a, G1a = self._partial_hold()
        final = None

        def value(knot):
            return wfull[knot] if 0 <= knot <= self.n else 0.0

        for k in range(self.n):
            feed = np.zeros(2)
            eta = self.delay_fraction
            for j, tap_delay in enumerate(DELAYS):
                delayed_value = (
                    eta * value(k - tap_delay - 1)
                    + (1.0 - eta) * value(k - tap_delay)
                )
                feed += f[:, j] * delayed_value
            u = -self.K @ xd + I0 * feed
            if k == self.n - 1:
                outputs = []
                for tau in np.asarray(phases, dtype=float):
                    At, Bt, G0t, G1t = foh_zoh_lift(
                        self.A, self.B, self.D, float(tau), self.h
                    )
                    xp = (
                        At @ x + Bt @ u
                        + G0t * (AMP * value(k))
                        + G1t * (AMP * value(k + 1))
                    )
                    lam = float(tau / self.h)
                    delta = AMP * (
                        (1.0 - lam) * value(k) + lam * value(k + 1)
                    )
                    outputs.append(self.C @ xp + self.Du @ u + self.Ddelta * delta)
                final = np.asarray(outputs)
                break
            xnext = (
                self.ad @ x + self.bd_phys @ u
                + self.g0 * (AMP * value(k))
                + self.g1 * (AMP * value(k + 1))
            )
            xdnext = (
                Aa @ x + Ba @ u
                + G0a * (AMP * value(k))
                + G1a * (AMP * value(k + 1))
            )
            x, xd = xnext, xdnext
        return final