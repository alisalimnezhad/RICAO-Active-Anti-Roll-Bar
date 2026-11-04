"""Simulate both cases from zero initial states using the same steering input."""
import numpy as np
from scipy.signal import lsim
import parameters as p
from vehicle_model import build_model
from lqr_controller import design_lqr

SIGNALS = ['passive_Rf','passive_Rr','LQR1_Rf','LQR1_Rr','uf_A','ur_A']


def rms(t, y):
    return np.sqrt(np.trapezoid(y*y,t,axis=0)/(t[-1]-t[0]))


def simulate(steering, dt=None):
    dt = p.DT if dt is None else dt
    if not np.isfinite(dt) or dt<=0 or dt>p.DURATION:
        raise ValueError('Sampling interval must be positive and <= duration.')
    steering = np.asarray(steering)
    if steering.ndim!=2 or steering.shape[1]!=2 or len(steering)<2 or not np.isfinite(steering).all():
        raise ValueError('Steering must have two finite columns: time_s,delta_rad.')
    if not np.all(np.diff(steering[:,0])>0):
        raise ValueError('Steering times must be strictly increasing.')
    t = np.linspace(0,p.DURATION,round(p.DURATION/dt)+1)
    delta = np.interp(t,steering[:,0],steering[:,1])
    active = build_model()
    passive = build_model(passive=True)
    K, Q, R, checks = design_lqr(active)
    closed = active['A']-active['B']@K
    xa = lsim((closed,active['D'],np.eye(10),np.zeros((10,1))),delta,t)[1]
    xp = lsim((passive['A'],passive['D'],np.eye(6),np.zeros((6,1))),delta,t)[1]
    signals = np.column_stack([xp@passive['C'].T,xa@active['C'].T,-xa@K.T])
    return dict(t=t,delta=delta,y=signals,xa=xa,xp=xp,K=K,Q=Q,R=R,
                closed=closed,active=active,passive=passive,checks=checks)