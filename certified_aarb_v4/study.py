"""Stage-4 operating-envelope utilities.

The steering/time certificates remain exact for every evaluated linear model.
Vehicle parameters are varied through a mechanically coupled payload family.
Continuous-parameter performance is assessed on a deterministic tensor grid
and an independent scrambled-Sobol holdout; it is not silently promoted to an
interval proof between parameter points.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "certified_aarb_v3" / "study.py"
_spec = importlib.util.spec_from_file_location("certified_aarb_v3_study", V3)
v3 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = v3
_spec.loader.exec_module(v3)

ParametricStudy = v3.ParametricStudy
RobustDesigner = v3.RobustDesigner
base_caps = v3.base_caps
base = v3.base
p = v3.p


def distributed_payload_overrides(mass_fraction, payload_height):
    """Mechanically coupled payload family used in version 4.

    A payload of mass ``mass_fraction * nominal sprung mass`` is placed at the
    nominal longitudinal CG and on the vehicle centreline.  Its own inertia per
    unit mass is taken equal to the nominal sprung body.  The combined sprung
    CG and roll inertia use centre-of-mass and parallel-axis relations.  When
    payload height equals the nominal height, the result is the proportional
    U10 case from version 3.
    """
    fraction = float(mass_fraction)
    height_payload = float(payload_height)
    if not (0.0 <= fraction <= 0.10 + 1e-14):
        raise ValueError("mass fraction outside declared interval")
    if not (p.h - 1e-14 <= height_payload <= 1.80 + 1e-14):
        raise ValueError("payload height outside declared interval")
    payload_mass = fraction * p.ms
    sprung_mass = p.ms + payload_mass
    combined_height = (
        p.ms * p.h + payload_mass * height_payload
    ) / sprung_mass
    roll_inertia = (
        p.Ixx * (1.0 + fraction)
        + p.ms * (p.h - combined_height) ** 2
        + payload_mass * (height_payload - combined_height) ** 2
    )
    return {
        "ms": sprung_mass,
        "m": sprung_mass + p.muf + p.mur,
        "h": combined_height,
        "Ixx": roll_inertia,
        "Ixz": p.Ixz * (1.0 + fraction),
        "Izz": p.Izz * (1.0 + fraction),
    }


def study_at(speed_kmh, mass_fraction, payload_height, feedback_gain,
             phases=9, label=None):
    return ParametricStudy(
        speed_kmh=float(speed_kmh),
        overrides=distributed_payload_overrides(mass_fraction, payload_height),
        feedback_gain=np.asarray(feedback_gain), phases=phases,
        label=label or (
            f"v={float(speed_kmh):.4g}; mu={float(mass_fraction):.4g}; "
            f"hp={float(payload_height):.4g}"
        ),
    )


def interpolate_schedule(speed_kmh, nodes, values):
    """Piecewise-linear interpolation with exact endpoint values."""
    speed = float(speed_kmh)
    nodes = np.asarray(nodes, dtype=float)
    if speed < nodes[0] - 1e-12 or speed > nodes[-1] + 1e-12:
        raise ValueError("speed outside schedule")
    if speed <= nodes[0]:
        return np.asarray(values[0], dtype=float).copy()
    if speed >= nodes[-1]:
        return np.asarray(values[-1], dtype=float).copy()
    right = int(np.searchsorted(nodes, speed, side="right"))
    left = right - 1
    alpha = (speed - nodes[left]) / (nodes[right] - nodes[left])
    return (1.0 - alpha) * np.asarray(values[left]) + alpha * np.asarray(values[right])


def compact_evaluate(study, coefficients, phases=21):
    ev = study.evaluate(np.asarray(coefficients, dtype=float), phases=phases)
    return {
        "continuous": dict(zip(study.names, ev["continuous_upper"])),
        "grid": dict(zip(study.names, ev["grid_upper"])),
        "interpolation": dict(zip(study.names, ev["interpolation_correction"])),
        "spectral_radius": study.rho_closed,
        "max_support_correction": float(np.max(ev["support_corrections"])),
    }


def selected_evaluate(study, coefficients, outputs=(0, 1), phases=21):
    """Evaluate selected output certificates with the same time remainder."""
    phase_times = np.linspace(0.0, study.h, int(phases))
    maps = study.maps(phase_times)
    g = study.combine_maps(maps, np.asarray(coefficients, dtype=float))
    selected = tuple(int(i) for i in outputs)
    grid = {}
    support_correction = 0.0
    for output in selected:
        values = []
        for phase in range(len(phase_times)):
            upper, _, _, correction = study.support(
                g[phase, output], return_witness=True
            )
            values.append(upper)
            support_correction = max(support_correction, float(correction))
        grid[study.names[output]] = float(max(values))
    spacing = study.h / (int(phases) - 1)
    interpolation, _, _ = study._intersample_correction(
        np.asarray(coefficients, dtype=float), spacing
    )
    continuous = {
        study.names[output]: grid[study.names[output]] + float(interpolation[output])
        for output in selected
    }
    return {
        "continuous": continuous,
        "grid": grid,
        "spectral_radius": study.rho_closed,
        "max_support_correction": support_correction,
    }


def design_skeleton(feedback_gain, phases=9):
    studies = []
    for speed in (50.0, 62.5, 75.0):
        for fraction, height, name in (
            (0.0, p.h, "nominal"),
            (0.10, p.h, "payload-low"),
            (0.10, 1.80, "payload-high"),
        ):
            studies.append(study_at(
                speed, fraction, height, feedback_gain, phases=phases,
                label=f"{speed:g} km/h; {name}",
            ))
    return studies