"""Recompute headline-defining records with 41/81 phase meshes."""
from __future__ import annotations

import json
from pathlib import Path

from study import evaluate_record


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def main() -> None:
    result_path = OUT / "results.json"
    result = json.loads(result_path.read_text())
    if result.get("status") != "PASS":
        raise RuntimeError("complete PASS screening results are required")
    by_id = {record["id"]: record for record in result["records"]}
    refined = {}
    for bank, summary in result["summaries"].items():
        identifiers = {
            "max_spectral_radius": summary["max_spectral_radius_id"],
            "worst_stable_ltr": summary["screen_worst_stable_ltr_id"],
            "max_utilization": summary["screen_max_utilization_id"],
            "min_rear_margin": summary["screen_min_rear_margin_id"],
        }
        records = {}
        for role, identifier in identifiers.items():
            source = by_id[identifier]
            records[role] = evaluate_record(
                identifier, source["unit"], source["radius"], 41, 81
            )
            if records[role]["error"] is not None:
                raise RuntimeError(records[role]["error"])
        refined[bank] = {"identifiers": identifiers, "records": records}
        summary.update({
            "refined_worst_stable_ltr": records["worst_stable_ltr"].get(
                "worst_ltr"
            ),
            "refined_max_utilization": records["max_utilization"].get(
                "max_utilization"
            ),
            "refined_min_rear_margin": records["min_rear_margin"].get(
                "rear_certificate", {}
            ).get("relative_margin"),
            "refined_min_rear_margin_percent": records["min_rear_margin"].get(
                "rear_certificate", {}
            ).get("relative_margin_percent"),
        })
    (OUT / "critical_refinement.json").write_text(
        json.dumps(refined, indent=2, sort_keys=True) + "\n"
    )
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary_path = OUT / "results_summary.json"
    summary_file = json.loads(summary_path.read_text())
    summary_file["summaries"] = result["summaries"]
    summary_path.write_text(json.dumps(summary_file, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
