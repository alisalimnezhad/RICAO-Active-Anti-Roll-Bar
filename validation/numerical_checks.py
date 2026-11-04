"""Solver checks and regression against the current manuscript's reported metrics."""
import numpy as np
from scipy.integrate import solve_ivp
from simulation import simulate, rms
from lqr_controller import STATE_SCALE


def verify(result, steering, expected):
    checks = dict(result['checks'])
    t,y = result['t'],result['y']
    fine = simulate(steering,dt=(t[1]-t[0])/2)
    checks['halved_step_max_relative_RMS_change'] = float(max(abs(rms(fine['t'],fine['y'])/rms(t,y)-1)))
    a = result['closed']*STATE_SCALE[None,:]/STATE_SCALE[:,None]
    d = result['active']['D'][:,0]/STATE_SCALE
    sol = solve_ivp(lambda tt,z:a@z+d*np.interp(tt,t,result['delta']),
                    (t[0],t[-1]),np.zeros(10),method='Radau',t_eval=t,
                    rtol=2e-9,atol=2e-11,jac=a,max_step=.02)
    if not sol.success: raise RuntimeError(sol.message)
    xa = sol.y.T*STATE_SCALE
    yr = np.column_stack([xa@result['active']['C'].T,-xa@result['K'].T])
    checks['radau_max_peak_normalized_error'] = float(max(np.max(abs(yr-y[:,2:]),axis=0)/np.max(abs(y[:,2:]),axis=0)))
    errors = abs(rms(t,y)/np.array(expected['RMS_SI'])-1)
    checks['manuscript_max_relative_RMS_error'] = float(max(errors))
    limits={'care_relative_residual':1e-7,'halved_step_max_relative_RMS_change':1e-3,
            'radau_max_peak_normalized_error':1e-4,'manuscript_max_relative_RMS_error':expected['rms_regression_rtol']}
    for name,bound in limits.items():
        if not np.isfinite(checks[name]) or checks[name]>=bound:
            raise AssertionError(f'{name}: {checks[name]} >= {bound}. If parameters were changed, the manuscript regression is no longer applicable.')
    if checks['largest_real_pole']>=0:
        raise AssertionError('LQR closed loop is not asymptotically stable.')
    checks['limits']=limits
    checks['status']='PASS: numerical checks and current-manuscript RMS regression'
    return checks