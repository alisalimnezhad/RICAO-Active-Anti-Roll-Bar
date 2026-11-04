"""Independent read-only verification of the saved stage-3 designs."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from study import (
    ParametricStudy, central_payload_overrides, proportional_mass_overrides,
    base,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def compact(study, f, phases):
    ev = study.evaluate(np.asarray(f, dtype=float), phases=phases)
    return {
        "continuous": dict(zip(study.names, ev["continuous_upper"])),
        "grid": dict(zip(study.names, ev["grid_upper"])),
        "interpolation": dict(zip(study.names, ev["interpolation_correction"])),
        "max_support_correction": float(np.max(ev["support_corrections"])),
    }


def make_bank(feedback_by_speed):
    loads = [
        ("nominal", {}),
        ("uniform-mass-plus-10pct", proportional_mass_overrides(0.10)),
        ("central-high-payload-plus-10pct", central_payload_overrides(0.10, 1.80)),
    ]
    bank = []
    for speed in (50.0, 70.0, 75.0):
        for load_name, overrides in loads:
            bank.append(ParametricStudy(
                speed_kmh=speed, overrides=overrides,
                feedback_gain=feedback_by_speed[speed], phases=9,
                label=f"{speed:g} km/h; {load_name}",
            ))
    return bank


def main():
    saved = json.loads((OUT / "stage3_results.json").read_text())
    nominal = ParametricStudy(speed_kmh=70.0, phases=9,
                              label="70 km/h; nominal")
    k70 = nominal.K.copy()
    zero = np.zeros(12)

    nominal_records = {}
    for name in ("pareto_0", "fair_current_balanced",
                 "fair_current_rear_protected"):
        record = json.loads((OUT / f"{name}.json").read_text())
        nominal_records[name] = compact(nominal, record["coefficients"], 81)
    nominal_baseline = compact(nominal, zero, 81)

    fixed_bank = make_bank({50.0: k70, 70.0: k70, 75.0: k70})
    fixed_balanced_f = json.loads((OUT / "bank_fixed_balanced.json").read_text())["coefficients"]
    fixed_protected_f = json.loads((OUT / "bank_fixed_rear_protected.json").read_text())["coefficients"]
    bank_rows = []
    for study in fixed_bank:
        bank_rows.append({
            "label": study.label,
            "baseline": compact(study, zero, 81),
            "fixed_balanced": compact(study, fixed_balanced_f, 81),
            "fixed_rear_protected": compact(study, fixed_protected_f, 81),
        })

    gains = {speed: ParametricStudy(speed_kmh=speed, phases=9).K.copy()
             for speed in (50.0, 70.0, 75.0)}
    scheduled_bank = make_bank(gains)
    scheduled_coefficients = {
        50.0: json.loads((OUT / "bank_scheduled_50p0.json").read_text())["coefficients"],
        70.0: json.loads((OUT / "bank_scheduled_70p0.json").read_text())["coefficients"],
        75.0: json.loads((OUT / "bank_scheduled_75p0.json").read_text())["coefficients"],
    }
    scheduled_rows = []
    for study in scheduled_bank:
        scheduled_rows.append({
            "label": study.label,
            "scheduled": compact(study, scheduled_coefficients[study.speed_kmh], 81),
        })

    tolerance = 2e-8
    nominal_rear_ok = bool(
        nominal_records["fair_current_rear_protected"]["continuous"]["rear_LTR"]
        <= nominal_baseline["continuous"]["rear_LTR"] + tolerance
    )
    bank_rear_ok = bool(all(
        row["fixed_rear_protected"]["continuous"]["rear_LTR"]
        <= row["baseline"]["continuous"]["rear_LTR"] + tolerance
        for row in bank_rows
    ))
    fixed_constraints_ok = bool(all(
        max(v for k, v in row[key]["continuous"].items()
            if k not in ("front_LTR", "rear_LTR")) <= 1.0 + tolerance
        for row in bank_rows for key in ("fixed_balanced", "fixed_rear_protected")
    ))
    scheduled_constraints_ok = bool(all(
        max(v for k, v in row["scheduled"]["continuous"].items()
            if k not in ("front_LTR", "rear_LTR")) <= 1.0 + tolerance
        for row in scheduled_rows
    ))
    fair = nominal_records["fair_current_rear_protected"]["continuous"]
    base_current = nominal_baseline["continuous"]
    fair_current_ok = bool(
        fair["front_current"] <= base_current["front_current"] + 3e-6
        and fair["rear_current"] <= base_current["rear_current"] + 3e-6
    )

    verification = {
        "status": "PASS" if all((nominal_rear_ok, bank_rear_ok,
                                  fixed_constraints_ok,
                                  scheduled_constraints_ok,
                                  fair_current_ok)) else "REVIEW",
        "phases": 81,
        "phase_spacing_ms": 0.25,
        "nominal_baseline": nominal_baseline,
        "nominal_designs": nominal_records,
        "bank": bank_rows,
        "scheduled_bank": scheduled_rows,
        "checks": {
            "nominal_rear_non_degradation": nominal_rear_ok,
            "bank_rear_non_degradation_all_9": bank_rear_ok,
            "fixed_bank_constraints_all_9": fixed_constraints_ok,
            "scheduled_bank_constraints_all_9": scheduled_constraints_ok,
            "equal_current_channel_caps": fair_current_ok,
        },
        "note": (
            "All output maps and steering-support LPs were rebuilt from the saved "
            "coefficients on an 81-phase certificate mesh."
        ),
    }
    base.save_json(OUT / "stage3_verification.json", verification)
    print(json.dumps({"status": verification["status"],
                      "checks": verification["checks"]}, indent=2))


if __name__ == "__main__":
    main()