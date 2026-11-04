"""Structural tests for the stage-3 parameter and fairness extensions."""
import unittest
import numpy as np

from study import (
    ParametricStudy, central_payload_overrides, proportional_mass_overrides,
    p,
)


class Stage3Tests(unittest.TestCase):
    def test_nominal_parameter_construction_is_unchanged(self):
        a = ParametricStudy(speed_kmh=70.0)
        b = ParametricStudy(speed_kmh=70.0, overrides={})
        np.testing.assert_allclose(a.A, b.A, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(a.K, b.K, rtol=0.0, atol=0.0)

    def test_temporary_parameters_are_restored(self):
        speed = p.v
        ParametricStudy(speed_kmh=50.0)
        self.assertEqual(p.v, speed)

    def test_uniform_mass_case_preserves_mass_balance(self):
        values = proportional_mass_overrides(0.10)
        self.assertAlmostEqual(values["m"], values["ms"] + p.muf + p.mur)
        self.assertAlmostEqual(values["Ixx"] / p.Ixx, 1.10)

    def test_central_payload_couples_mass_height_and_roll_inertia(self):
        values = central_payload_overrides(0.10, 1.80)
        self.assertGreater(values["ms"], p.ms)
        self.assertGreater(values["h"], p.h)
        self.assertLess(values["h"], 1.80)
        self.assertGreater(values["Ixx"], p.Ixx)
        self.assertAlmostEqual(values["m"], values["ms"] + p.muf + p.mur)

    def test_fixed_gain_is_retained_across_speed(self):
        nominal = ParametricStudy(speed_kmh=70.0)
        changed = ParametricStudy(speed_kmh=50.0, feedback_gain=nominal.K)
        np.testing.assert_allclose(changed.K, nominal.K, rtol=0.0, atol=0.0)
        self.assertFalse(np.allclose(changed.A, nominal.A))


if __name__ == "__main__":
    unittest.main(verbosity=2)