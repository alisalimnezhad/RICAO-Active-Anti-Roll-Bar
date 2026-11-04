"""Write protocol metadata, source hashes and the release SHA-256 manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from study import BANKS, STRESS_DIMENSION


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result_path = OUT / "results.json"
    summary_path = OUT / "results_summary.json"
    verification_path = OUT / "independent_verification.json"
    for path in (result_path, summary_path, verification_path):
        if not path.exists():
            raise FileNotFoundError(f"required completed output is missing: {path}")

    source_paths = [
        HERE / name for name in (
            "study.py", "run_all.py", "run_shard.py",
            "merge_shards.py",
            "refine_critical.py", "independent_verify.py", "make_artifacts.py",
            "finalize_archive.py", "test_robustness.py",
        )
    ] + [
        ROOT / "certified_aarb_v4" / "study.py",
        ROOT / "certified_aarb_v3" / "study.py",
        ROOT / "certified_aarb_v2" / "study.py",
        ROOT / "validation" / "vehicle_model.py",
        ROOT / "validation" / "parameters.py",
        ROOT / "validation" / "lqr_controller.py",
        ROOT / "certified_aarb_v5_rear_audit" / "outputs" /
        "reserve_5pct" / "coefficients.csv",
    ]
    missing = [str(path) for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing sources:\n" + "\n".join(missing))

    protocol = {
        "date": "2026-09-27",
        "dimension": STRESS_DIMENSION,
        "banks": BANKS,
        "output_phases": 21,
        "rear_phases": 41,
        "critical_output_phases": 41,
        "critical_rear_phases": 81,
        "execution": {
            "method": "two independent resumable shards",
            "shard_count": 2,
            "record_count": 1536,
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "solver": "scipy.optimize.linprog/HiGHS with certified fallbacks",
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )

    manifest = {}
    for path in sorted(
        item for item in HERE.rglob("*")
        if item.is_file()
        and item.name != "hash_manifest.json"
        and not item.name.startswith("records_shard_")
        and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
    ):
        manifest[str(path.relative_to(ROOT))] = sha256(path)
    for path in (
        ROOT / "paper" / "Main_Manuscript.tex",
        ROOT / "paper" / "Supplementary_Material.tex",
        ROOT / "paper" / "tables" / "table_combined_robustness_summary.tex",
        ROOT / "paper" / "figures" / "combined_robustness_summary.png",
    ):
        if path.exists():
            manifest[str(path.relative_to(ROOT))] = sha256(path)
    (OUT / "hash_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"manifest_entries": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()