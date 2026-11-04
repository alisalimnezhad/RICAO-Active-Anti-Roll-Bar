"""Read-only second-pass verification of the saved 20 mA reproduction."""
from pathlib import Path
import hashlib
import json

import numpy as np

from study import Study, save_json

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def main():
    result = json.loads((OUT / "reproduction_20mA.json").read_text())
    f = np.asarray(result["controller_coefficients"]).reshape(-1)
    csv = np.loadtxt(OUT / "controller_coefficients_20mA.csv", delimiter=",", skiprows=1)
    if not np.allclose(csv[:, 2:].T.reshape(-1), f, rtol=0.0, atol=1e-14):
        raise AssertionError("coefficient CSV and JSON disagree")
    source_hash = hashlib.sha256((HERE / "study.py").read_bytes()).hexdigest()
    if source_hash != result["study_source_sha256"]:
        raise AssertionError("study source changed after the saved run")

    study = Study(phases=9)
    old_refinement = {
        5: 0.0630912, 9: 0.0502302, 21: 0.0466291,
        41: 0.0461148, 81: 0.0459867,
    }
    refinement = {}
    for phases, old in old_refinement.items():
        ev = study.evaluate(f, phases=phases)
        upper = float(np.max(ev["continuous_upper"][:2]))
        constraints = ev["continuous_upper"][2:]
        refinement[str(phases)] = {
            "new_upper": upper, "reported_v1_upper": old,
            "absolute_difference": abs(upper - old),
            "max_constraint_utilization": float(np.max(constraints)),
            "feasible": bool(np.max(constraints) <= 1.0 + 2e-8),
        }
    saved = float(result["new_results"]["proposed_upper"])
    if abs(refinement["41"]["new_upper"] - saved) > 2e-10:
        raise AssertionError("saved and recomputed 41-phase bounds disagree")

    archive = np.load(OUT / "audit_witness.npz")
    direct_max = np.max(np.abs(archive["outputs"]), axis=0)
    certified = np.asarray([result["continuous_output_bounds"][name] for name in study.names])
    if np.max(direct_max - certified) > 2e-8:
        raise AssertionError("archived direct response exceeds the certificate")

    verification = {
        "status": "PASS",
        "coefficient_files_agree": True,
        "source_hash_matches_saved_run": True,
        "refinement": refinement,
        "direct_witness_max_excess": float(np.max(direct_max - certified)),
        "all_41_phase_constraints_feasible": refinement["41"]["feasible"],
        "note": "The five-mesh comparison reuses the independently saved coefficients but rebuilds all lifted maps and support LPs.",
    }
    save_json(OUT / "verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()