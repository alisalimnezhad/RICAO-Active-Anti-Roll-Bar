"""Scalar residuals of paper Eqs. (10)--(13),(21)--(24), independently assembled.
Active states: beta, yaw_rate, phi, roll_rate, phi_uf, phi_ur, Pf, Xvf, Pr, Xvr.
Passive states: the first six, with nonzero mechanical ARB stiffness and no hydraulics.
"""
import numpy as np
import parameters as p

STATE_NAMES = ['beta_rad','yaw_rate_rad_s','phi_rad','roll_rate_rad_s',
               'phi_uf_rad','phi_ur_rad','Pf_Pa','Xvf_m','Pr_Pa','Xvr_m']


def residual(x, dx, delta, current, passive=False, lact=None):
    lact = p.L_ACT if lact is None else lact
    beta, yaw, phi, roll, uf, ur = x[:6]
    db, dy, dp, dr, duf, dur = dx[:6]
    fyf = p.mu*p.Cf*(-beta+delta-p.lf*yaw/p.v)
    fyr = p.mu*p.Cr*(-beta+p.lr*yaw/p.v)
    sf = p.kf*(phi-uf)+p.bf*(roll-duf)
    sr = p.kr*(phi-ur)+p.br*(roll-dur)
    if passive:
        tf = 4*p.kAOf/p.ARM_LENGTH**2*(p.T_A*p.T_B*phi-p.T_A**2*uf)
        tr = 4*p.kAOr/p.ARM_LENGTH**2*(p.T_A*p.T_B*phi-p.T_A**2*ur)
    else:
        tf = 2*lact*p.area*x[6]
        tr = 2*lact*p.area*x[8]
    equations = [
        p.m*p.v*(db+yaw)-p.ms*p.h*dr-fyf-fyr,
        -p.Ixz*dr+p.Izz*dy-p.lf*fyf+p.lr*fyr,
        (p.Ixx+p.ms*p.h*p.h)*dr-p.Ixz*dy-p.ms*p.g*p.h*phi
        -p.ms*p.v*p.h*(db+yaw)+sf+sr-tf-tr,
        -p.rh*fyf-p.muf*p.v*(p.rh-p.hu)*(db+yaw)-p.muf*p.g*p.hu*uf+p.ktf*uf-sf-tf,
        # Retain the rear gravity sign as printed in Eq. (10) and the Appendix.
        -p.rh*fyr-p.mur*p.v*(p.rh-p.hu)*(db+yaw)+p.mur*p.g*p.hu*ur+p.ktr*ur-sr-tr,
        dp-roll,
    ]
    if not passive:
        pf, xf, pr, xr = x[6:]
        dpf, dxf, dpr, dxr = dx[6:]
        equations.extend([
            p.Vt/(4*p.bulk)*dpf+(p.KP+p.Ctp)*pf-p.Kx*xf+p.area*lact*(roll-duf),
            dxf+xf/p.tau-p.Kv/p.tau*current[0],
            p.Vt/(4*p.bulk)*dpr+(p.KP+p.Ctp)*pr-p.Kx*xr+p.area*lact*(roll-dur),
            dxr+xr/p.tau-p.Kv/p.tau*current[1],
        ])
    return np.array(equations)


def build_model(passive=False, lact=None):
    n = 6 if passive else 10
    zero = np.zeros(n)
    def f(x, dx, delta=0., current=None):
        if current is None:
            current = np.zeros(2)
        return residual(x, dx, delta, current, passive, lact)
    E = np.column_stack([f(zero,e) for e in np.eye(n)])
    F = -np.column_stack([f(e,zero) for e in np.eye(n)])
    G = -f(zero,zero,delta=1.)[:,None]
    H = -np.column_stack([f(zero,zero,current=e) for e in np.eye(2)])
    C = np.zeros((2,n))
    C[0,4] = p.ktf/p.lw/(p.m*p.lr/(p.lf+p.lr))/p.g
    C[1,5] = p.ktr/p.lw/(p.m*p.lf/(p.lf+p.lr))/p.g
    return dict(E=E, F=F, G=G, H=H, A=np.linalg.solve(E,F),
                B=np.linalg.solve(E,H), D=np.linalg.solve(E,G), C=C)