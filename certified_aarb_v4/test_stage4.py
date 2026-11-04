import unittest
import numpy as np

from study import distributed_payload_overrides, interpolate_schedule, p


class Stage4Tests(unittest.TestCase):
    def test_zero_payload_returns_nominal_mass_properties(self):
        q = distributed_payload_overrides(0.0, 1.80)
        self.assertAlmostEqual(q["ms"], p.ms)
        self.assertAlmostEqual(q["m"], p.m)
        self.assertAlmostEqual(q["h"], p.h)
        self.assertAlmostEqual(q["Ixx"], p.Ixx)
        self.assertAlmostEqual(q["Ixz"], p.Ixz)
        self.assertAlmostEqual(q["Izz"], p.Izz)

    def test_low_payload_reproduces_proportional_case(self):
        q = distributed_payload_overrides(0.10, p.h)
        self.assertAlmostEqual(q["ms"] / p.ms, 1.10)
        self.assertAlmostEqual(q["Ixx"] / p.Ixx, 1.10)
        self.assertAlmostEqual(q["Ixz"] / p.Ixz, 1.10)
        self.assertAlmostEqual(q["Izz"] / p.Izz, 1.10)

    def test_high_payload_couples_cg_and_inertia(self):
        low = distributed_payload_overrides(0.10, p.h)
        high = distributed_payload_overrides(0.10, 1.80)
        self.assertGreater(high["h"], low["h"])
        self.assertGreater(high["Ixx"], low["Ixx"])
        self.assertAlmostEqual(high["m"], high["ms"] + p.muf + p.mur)

    def test_schedule_endpoints_and_midpoint(self):
        nodes = [50.0, 70.0, 75.0]
        values = [np.array([0.0, 2.0]), np.array([2.0, 4.0]), np.array([4.0, 8.0])]
        np.testing.assert_allclose(interpolate_schedule(50, nodes, values), values[0])
        np.testing.assert_allclose(interpolate_schedule(75, nodes, values), values[2])
        np.testing.assert_allclose(interpolate_schedule(60, nodes, values), [1.0, 3.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)