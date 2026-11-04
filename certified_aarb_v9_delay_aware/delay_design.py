"""Utilities for the independently redesigned 5 ms delayed controller."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUDIT = ROOT / "certified_aarb_v5_rear_audit"
V4 = ROOT / "certified_aarb_v4"
for path in (AUDIT, V4):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from delay_study import DelayedParametricStudy  # noqa: E402
from study import (  # noqa: E402
    RobustDesigner,
    base,
    base_caps,
    compact_evaluate,
    distributed_payload_overrides,
    p,
)
from run_envelope import controller_summary, points  # noqa: E402


DELAY_S = 0.005


def delayed_study(point, gain, phases=5, label=None):
    return DelayedParametricStudy(
        speed_kmh=float(point["speed_kmh"]),
        overrides=distributed_payload_overrides(
            point["mass_fraction"], point["payload_height"]
        ),
        feedback_gain=np.asarray(gain, dtype=float),
        delay_s=DELAY_S,
        phases=int(phases),
        label=label or (
            f"5 ms; v={point['speed_kmh']:.6g}; "
            f"mu={point['mass_fraction']:.6g}; hp={point['payload_height']:.6g}"
        ),
    )


def design_points(corners_only=False):
    speeds = (50.0, 75.0) if corners_only else (50.0, 62.5, 75.0)
    loads = (
        (0.0, p.h, "nominal"),
        (0.10, 1.80, "payload-high"),
    ) if corners_only else (
        (0.0, p.h, "nominal"),
        (0.10, p.h, "payload-low"),
        (0.10, 1.80, "payload-high"),
    )
    return [
        {
            "set": "design",
            "speed_kmh": speed,
            "mass_fraction": fraction,
            "payload_height": height,
            "label": f"{speed:g} km/h; {name}",
        }
        for speed in speeds
        for fraction, height, name in loads
    ]


def build_design_bank(gain, corners_only=False, phases=5):
    return [
        delayed_study(point, gain, phases=phases, label=point["label"])
        for point in design_points(corners_only=corners_only)
    ]


def delayed_radius(study, gain):
    Aa, Ba, _, _ = study._partial_hold()
    matrix = np.block([
        [study.ad, -study.bd_phys @ gain],
        [Aa, -Ba @ gain],
    ])
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))