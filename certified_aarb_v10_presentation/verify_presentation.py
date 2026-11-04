"""Verify provenance and consistency of the shipped presentation assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    data = json.loads((OUT / "presentation_data.json").read_text())
    controllers = {row["controller"]: row for row in data["controllers"]}
    artifact_paths = (
        Path("figures/certification_architecture.png"),
        Path("tables/certificate_scope.tex"),
        Path("tables/complexity_implementation.tex"),
        Path("tables/complexity_performance.tex"),
    )
    checks = {
        "ten_scope_rows": len(data["scope_rows"]) == 10,
        "seven_controller_rows": len(controllers) == 7,
        "delay_aware_value_matches": abs(
            controllers["5 ms delay-aware"]["worst_ltr"] - 0.046692792912060445
        ) < 1e-14,
        "scheduled_storage_exceeds_fixed": (
            controllers["Scheduled balanced"]["stored"] >
            controllers["Rear protected"]["stored"]
        ),
        "fir4_uses_42_points": controllers["Four-tap reserve"]["coverage"] == "42 structured",
        "microbenchmark_disclaimed": "not ECU" in data["benchmark"]["environment"],
        "source_hashes_match": all(
            sha256(ROOT / relative) == expected
            for relative, expected in data["source_hashes"].items()
        ),
        "artifact_assets_present": all(
            (ROOT / "paper" / relative).is_file()
            and (ROOT / "paper" / relative).stat().st_size > 0
            for relative in artifact_paths
        ),
    }
    result = {"date": "2026-09-23", "status": "PASS" if all(checks.values()) else "FAIL",
              "checks": checks}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()