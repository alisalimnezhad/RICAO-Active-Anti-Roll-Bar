"""Nonlinear screening surrogate for critical delayed-controller cases.

This model deliberately remains separate from the linear certificate.  It
retains the published inertial and hydraulic residual structure while adding
smooth load-sensitive tyre saturation, current/spool saturation, pressure
relief and cubic suspension stops.  Its Jacobian at the origin is required to
match the declared linear plant before any critical-case result is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np

from delay_design import base


P = base.p
STATE_SCALE = base.STATE_SCALE


@contextmanager
def temporary_parameters(overrides):
    saved = {name: getattr(P, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(P, name, float(value))
        yield
    finally:
        for name, value in saved.items():
            setattr(P, name, value)


@dataclass
class NonlinearPlant:
    study: object

    def __post_init__(self):
        names = (
            "m", "ms", "muf", "mur", "lf", "lr", "lw", "h", "rh", "hu",
            "v", "g", "Ixx", "Ixz", "Izz", "kf", "kr", "ktf", "ktr",
            "bf", "br", "Cf", "Cr", "mu", "area", "Kx", "KP", "Vt",
            "bulk", "tau", "Kv", "Ctp", "L_ACT",
        )
        with temporary_parameters(self.study.parameter_overrides):
            self.par = {name: float(getattr(P, name)) for name in names}
        self.E = np.asarray(self.study.model["E"], dtype=float)
        self.front_static_load = (
            self.par["m"] * self.par["g"] * self.par["lr"] /
            (self.par["lf"] + self.par["lr"])
        )
        self.rear_static_load = (
            self.par["m"] * self.par["g"] * self.par["lf"] /
            (self.par["lf"] + self.par["lr"])
        )
        self.spool_limit = 0.000485
        self.pressure_relief = 9.5e6
        self.stop_angle = np.deg2rad(5.0)
        self.stop_gain = 2.0e5 / np.deg2rad(2.0) ** 3

    def load_transfer(self, x):
        front = self.par["ktf"] * x[4] / (
            self.par["lw"] * self.front_static_load
        )
        rear = self.par["ktr"] * x[5] / (
            self.par["lw"] * self.rear_static_load
        )
        return float(front), float(rear)

    @staticmethod
    def _stop(relative, threshold, gain):
        excess = max(abs(float(relative)) - threshold, 0.0)
        return np.sign(relative) * gain * excess ** 3

    @staticmethod
    def _relief(pressure, threshold, conductance):
        excess = max(abs(float(pressure)) - threshold, 0.0)
        return np.sign(pressure) * conductance * excess

    def tyre_forces(self, x, delta):
        q = self.par
        beta, yaw = x[0], x[1]
        alpha_f = -beta + delta - q["lf"] * yaw / q["v"]
        alpha_r = -beta + q["lr"] * yaw / q["v"]
        rf, rr = self.load_transfer(x)
        # The capacity falls as one side unloads.  The square-root form has
        # zero first derivative at the equilibrium, preserving the Jacobian.
        front_factor = np.sqrt(max(0.04, 1.0 - min(abs(rf), 0.999) ** 2))
        rear_factor = np.sqrt(max(0.04, 1.0 - min(abs(rr), 0.999) ** 2))
        cap_f = q["mu"] * self.front_static_load * front_factor
        cap_r = q["mu"] * self.rear_static_load * rear_factor
        fyf = cap_f * np.tanh(q["Cf"] * alpha_f / cap_f)
        fyr = cap_r * np.tanh(q["Cr"] * alpha_r / cap_r)
        return float(fyf), float(fyr), float(cap_f), float(cap_r)

    def rhs(self, x, delta, current):
        q = self.par
        x = np.asarray(x, dtype=float)
        current = np.clip(np.asarray(current, dtype=float), -0.02, 0.02)
        beta, yaw, phi, roll, uf, ur, pf, xf, pr, xr = x
        fyf, fyr, _, _ = self.tyre_forces(x, float(delta))
        sf = q["kf"] * (phi - uf) + q["bf"] * roll
        sr = q["kr"] * (phi - ur) + q["br"] * roll
        sf += self._stop(phi - uf, self.stop_angle, self.stop_gain)
        sr += self._stop(phi - ur, self.stop_angle, self.stop_gain)
        tf = 2.0 * q["L_ACT"] * q["area"] * pf
        tr = 2.0 * q["L_ACT"] * q["area"] * pr
        xfe = self.spool_limit * np.tanh(xf / self.spool_limit)
        xre = self.spool_limit * np.tanh(xr / self.spool_limit)
        relief_f = self._relief(pf, self.pressure_relief, 4.0 * q["KP"])
        relief_r = self._relief(pr, self.pressure_relief, 4.0 * q["KP"])
        residual = np.array([
            q["m"] * q["v"] * yaw - fyf - fyr,
            -q["lf"] * fyf + q["lr"] * fyr,
            -q["ms"] * q["g"] * q["h"] * phi
            - q["ms"] * q["v"] * q["h"] * yaw + sf + sr - tf - tr,
            -q["rh"] * fyf - q["muf"] * q["v"] * (q["rh"] - q["hu"]) * yaw
            - q["muf"] * q["g"] * q["hu"] * uf + q["ktf"] * uf - sf - tf,
            -q["rh"] * fyr - q["mur"] * q["v"] * (q["rh"] - q["hu"]) * yaw
            + q["mur"] * q["g"] * q["hu"] * ur + q["ktr"] * ur - sr - tr,
            -roll,
            (q["KP"] + q["Ctp"]) * pf + relief_f - q["Kx"] * xfe
            + q["area"] * q["L_ACT"] * roll,
            xf / q["tau"] - q["Kv"] / q["tau"] * current[0],
            (q["KP"] + q["Ctp"]) * pr + relief_r - q["Kx"] * xre
            + q["area"] * q["L_ACT"] * roll,
            xr / q["tau"] - q["Kv"] / q["tau"] * current[1],
        ])
        return -np.linalg.solve(self.E, residual)

    def jacobian_audit(self):
        zero = np.zeros(10)
        A = np.zeros((10, 10))
        for j, scale in enumerate(STATE_SCALE):
            step = max(1e-10, 2e-6 * float(scale))
            plus = zero.copy(); plus[j] += step
            minus = zero.copy(); minus[j] -= step
            A[:, j] = (
                self.rhs(plus, 0.0, np.zeros(2))
                - self.rhs(minus, 0.0, np.zeros(2))
            ) / (2.0 * step)
        B = np.zeros((10, 2))
        for j in range(2):
            step = 1e-7
            plus = np.zeros(2); plus[j] = step
            minus = np.zeros(2); minus[j] = -step
            B[:, j] = (
                self.rhs(zero, 0.0, plus) - self.rhs(zero, 0.0, minus)
            ) / (2.0 * step)
        step = 1e-7
        D = (
            self.rhs(zero, step, np.zeros(2))
            - self.rhs(zero, -step, np.zeros(2))
        )[:, None] / (2.0 * step)
        def rel(actual, reference):
            return float(np.linalg.norm(actual - reference) /
                         max(np.linalg.norm(reference), 1e-30))
        return {
            "relative_A": rel(A, self.study.A),
            "relative_B": rel(B, self.study.B),
            "relative_D": rel(D, self.study.D),
            "max_abs_A": float(np.max(np.abs(A - self.study.A))),
            "max_abs_B": float(np.max(np.abs(B - self.study.B))),
            "max_abs_D": float(np.max(np.abs(D - self.study.D))),
        }


def steering_value(w, time_s, h=0.02):
    if time_s <= 0.0:
        return 0.0
    wfull = np.r_[0.0, np.asarray(w, dtype=float)]
    if time_s >= h * (len(wfull) - 1):
        return float(wfull[-1])
    position = time_s / h
    left = int(np.floor(position))
    fraction = position - left
    return float((1.0 - fraction) * wfull[left] + fraction * wfull[left + 1])


def simulate(study, coefficients, steering, nonlinear, dt=0.0005):
    h = float(study.h)
    delay = 0.005
    if abs(round(h / dt) * dt - h) > 1e-12 or abs(round(delay / dt) * dt - delay) > 1e-12:
        raise ValueError("dt must divide the controller period and delay")
    coefficients = np.asarray(coefficients, dtype=float).reshape(2, 6)
    delays = np.asarray(base.DELAYS, dtype=int)
    steps = int(round(study.n * h / dt))
    hold_steps = int(round(h / dt))
    delay_steps = int(round(delay / dt))
    time = np.arange(steps + 1, dtype=float) * dt
    state = np.zeros((steps + 1, 10))
    current = np.zeros((steps + 1, 2))
    raw_current = np.zeros_like(current)
    steering_rad = np.zeros(steps + 1)
    plant = NonlinearPlant(study) if nonlinear else None
    held = np.zeros(2)

    def rhs(x, t, u):
        delta = base.AMP * steering_value(steering, t, h)
        if nonlinear:
            return plant.rhs(x, delta, u)
        return study.A @ x + study.B @ u + study.D[:, 0] * delta

    for i in range(steps):
        t = time[i]
        if i % hold_steps == 0:
            delayed_index = max(0, i - delay_steps)
            feed = np.zeros(2)
            for j, tap in enumerate(delays):
                feed += coefficients[:, j] * steering_value(
                    steering, t - delay - tap * h, h
                )
            raw = -study.K @ state[delayed_index] + base.I0 * feed
            held = np.clip(raw, -base.I0, base.I0) if nonlinear else raw
        raw_current[i] = raw
        current[i] = held
        steering_rad[i] = base.AMP * steering_value(steering, t, h)
        x = state[i]
        k1 = rhs(x, t, held)
        k2 = rhs(x + 0.5 * dt * k1, t + 0.5 * dt, held)
        k3 = rhs(x + 0.5 * dt * k2, t + 0.5 * dt, held)
        k4 = rhs(x + dt * k3, t + dt, held)
        state[i + 1] = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    current[-1] = current[-2]
    raw_current[-1] = raw_current[-2]
    steering_rad[-1] = base.AMP * steering_value(steering, time[-1], h)
    outputs = state @ study.C.T + current @ study.Du.T
    outputs += steering_rad[:, None] * study.Ddelta[None, :]
    return {
        "time": time,
        "state": state,
        "current": current,
        "raw_current": raw_current,
        "steering": steering_rad,
        "outputs": outputs,
        "plant": plant,
    }