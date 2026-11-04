"""Optional fast tests: python test_validation.py"""
import unittest
from pathlib import Path
import json
import numpy as np
import parameters as p
from vehicle_model import build_model
from lqr_controller import design_lqr
from simulation import simulate, rms

BASE=Path(__file__).parent

class ValidationTests(unittest.TestCase):
    def test_mass_balance(self):
        self.assertAlmostEqual(p.ms+p.muf+p.mur,p.m)

    def test_dimensions(self):
        self.assertEqual(build_model()['A'].shape,(10,10))
        self.assertEqual(build_model(True)['A'].shape,(6,6))

    def test_roll_kinematics(self):
        a=build_model()['A']
        e=np.zeros(10);e[3]=1
        np.testing.assert_allclose(a[2],e,atol=1e-12)

    def test_spool_equations(self):
        model=build_model()
        for row,col in [(7,0),(9,1)]:
            self.assertAlmostEqual(model['A'][row,row],-1/p.tau)
            self.assertAlmostEqual(model['B'][row,col],p.Kv/p.tau)

    def test_torque_factor(self):
        f=build_model()['F']
        self.assertAlmostEqual(f[2,6],2*p.L_ACT*p.area)
        self.assertAlmostEqual(f[2,8],2*p.L_ACT*p.area)

    def test_passive_independent_of_actuator_spacing(self):
        np.testing.assert_array_equal(build_model(True,lact=.5)['A'],build_model(True,lact=.93)['A'])

    def test_lqr(self):
        k,q,r,checks=design_lqr(build_model())
        self.assertEqual(k.shape,(2,10))
        np.testing.assert_array_equal(r,np.eye(2))
        self.assertLess(checks['care_relative_residual'],1e-7)
        self.assertLess(checks['largest_real_pole'],0)

    def test_data(self):
        for filename in (BASE/'data').glob('*.csv'):
            a=np.loadtxt(filename,delimiter=',',skiprows=1)
            self.assertEqual(a.shape[1],2)
            self.assertTrue(np.isfinite(a).all())
            self.assertTrue((np.diff(a[:,0])>0).all())

    def test_zero_steering(self):
        r=simulate(np.array([[0.,0.],[10.,0.]]))
        np.testing.assert_array_equal(r['y'],np.zeros_like(r['y']))

    def test_manuscript_rms(self):
        s=np.loadtxt(BASE/'data/steering.csv',delimiter=',',skiprows=1)
        expected=json.loads((BASE/'data/expected_metrics.json').read_text())
        r=simulate(s)
        np.testing.assert_allclose(rms(r['t'],r['y']),expected['RMS_SI'],rtol=expected['rms_regression_rtol'],atol=0)

if __name__=='__main__': unittest.main(verbosity=2)