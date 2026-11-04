"""Nominal LQR1, Eq. (33): five output weights=1, R=I2, u=-Kx in amperes."""
import numpy as np
from scipy.linalg import solve_continuous_are

STATE_SCALE = np.array([.1,1.,.1,1.,.1,.1,1e7,1e-3,1e7,1e-3])


def design_lqr(model):
    A, B, C = model['A'], model['B'], model['C']
    L = np.zeros((5,10))
    L[0,2] = 1
    L[1:3] = C
    L[3,[2,4]] = [1,-1]
    L[4,[2,5]] = [1,-1]
    Q = L.T@L
    R = np.eye(2)
    # Coordinate scaling only: this does not normalize or alter the physical cost.
    scale = np.diag(STATE_SCALE)
    a = np.linalg.solve(scale,A@scale)
    b = np.linalg.solve(scale,B)
    q = scale.T@Q@scale
    P = solve_continuous_are(a,b,q,R)
    K = np.linalg.solve(R,b.T@P)@np.linalg.inv(scale)
    terms = [a.T@P,P@a,-P@b@np.linalg.solve(R,b.T@P),q]
    checks = {
        'care_relative_residual':float(np.linalg.norm(sum(terms))/sum(np.linalg.norm(x) for x in terms)),
        'largest_real_pole':float(np.linalg.eigvals(A-B@K).real.max()),
    }
    return K, Q, R, checks