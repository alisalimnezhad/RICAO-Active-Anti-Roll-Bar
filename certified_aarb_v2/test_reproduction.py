"""Independent structural tests for the reconstructed certificate code."""
import unittest
import numpy as np
from scipy.linalg import expm

from study import Study, foh_zoh_lift, AMP, RHO


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Study(phases=3, n=20)

    def test_exact_lift_against_augmented_exponential(self):
        s = self.s
        rng = np.random.default_rng(4)
        x, u = rng.normal(size=10), rng.normal(size=2)
        d0, d1 = rng.normal(size=2)
        Ad, Bd, G0, G1 = foh_zoh_lift(s.A, s.B, s.D, s.h)
        left = Ad @ x + Bd @ u + G0 * d0 + G1 * d1
        n = 10
        M = np.zeros((n + 4, n + 4))
        M[:n, :n], M[:n, n:n+2], M[:n, n+2] = s.A, s.B, s.D[:, 0]
        M[n+2, n+3] = 1.0
        z = np.r_[x, u, d0, (d1-d0)/s.h]
        right = (expm(M*s.h) @ z)[:n]
        scaled = np.max(np.abs(left-right) / np.maximum(1.0, np.abs(right)))
        self.assertLess(scaled, 2e-12)

    def test_steering_witness_is_admissible(self):
        s = self.s
        g = np.linspace(-1, 2, s.n)
        _, w, _, _ = s.support(g, return_witness=True)
        self.assertLessEqual(np.max(np.abs(w)), 1+1e-10)
        self.assertLessEqual(np.max(np.abs(np.diff(np.r_[0.0,w]))), RHO+1e-10)

    def test_support_against_small_vertex_grid(self):
        s = Study(phases=2, n=3)
        g = np.array([0.7,-0.2,1.1])
        upper, _, primal, correction = s.support(g, return_witness=True)
        vals = np.arange(-1, 1.0001, RHO)
        feasible=[]
        for a in vals:
            for b in vals:
                for c in vals:
                    w=np.array([a,b,c])
                    if np.max(np.abs(np.diff(np.r_[0,w])))<=RHO+1e-12:
                        feasible.append(g@w)
        self.assertGreaterEqual(upper+1e-11, max(feasible))
        self.assertAlmostEqual(upper, primal, places=8)
        self.assertLess(correction, 1e-8)

    def test_controller_has_no_future_steering(self):
        s = self.s
        rng=np.random.default_rng(8)
        f=rng.normal(size=12)
        w1=np.zeros(s.n);w2=np.zeros(s.n)
        w2[10:]=0.1
        # Simulations agree before the first changed knot.
        u1=s.simulate(f,w1,refine=1)["current"]
        u2=s.simulate(f,w2,refine=1)["current"]
        self.assertLess(np.max(np.abs(u1[:10]-u2[:10])),1e-13)

    def test_output_units_at_zero(self):
        s=self.s
        tr=s.simulate(np.zeros(12),np.zeros(s.n),refine=1)
        self.assertEqual(float(np.max(np.abs(s.output_history(tr)))),0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)