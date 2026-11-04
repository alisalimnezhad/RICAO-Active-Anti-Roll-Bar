"""Final consistency gate for solver-derived v14 robustness material."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
ROBUST = ROOT / "certified_aarb_v14_combined_robustness"
ROUT = ROBUST / "outputs"
MAIN = ROOT / "paper" / "Main_Manuscript.tex"
SUPP = ROOT / "paper" / "Supplementary_Material.tex"
TABLE = ROOT / "paper" / "tables" / "table_combined_robustness_summary.tex"
FIGURE = ROOT / "paper" / "figures" / "combined_robustness_summary.png"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_table(summaries):
    labels = {"pm5": r"$\pm5\%$", "pm10": r"$\pm10\%$"}
    rows = []
    for bank in ("pm5", "pm10"):
        summary = summaries[bank]
        rows.append(
            f"{labels[bank]} & {summary['sample_count']} & "
            f"{summary['proposed_stable_count']} & {summary['feasible_count']} & "
            f"{summary['rear_pass_count']} & {summary['full_pass_count']} & "
            f"{summary['refined_worst_stable_ltr']:.6f} & "
            f"{summary['refined_max_utilization']:.4f} & "
            f"{summary['refined_min_rear_margin_percent']:.3f} & "
            f"{summary['max_spectral_radius']:.4f} \\\\"
        )
    return "\n".join(rows) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads((ROUT / "results.json").read_text())
    summary_file = json.loads((ROUT / "results_summary.json").read_text())
    verification = json.loads((ROUT / "independent_verification.json").read_text())
    summaries = summary_file["summaries"]
    pm5 = summaries["pm5"]
    pm10 = summaries["pm10"]
    records = results["records"]
    ids = [record["id"] for record in records]
    main_text = MAIN.read_text()
    supp_text = SUPP.read_text()
    table_text = TABLE.read_text()
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "test_robustness.py"],
        cwd=ROBUST, capture_output=True, text=True,
    )
    checks = {
        "sample_count_1536": len(records) == 1536,
        "unique_ids_1536": len(set(ids)) == 1536,
        "all_assemblies_finite": all(record["finite"] for record in records),
        "no_record_errors": all(record["error"] is None for record in records),
        "all_attempted_support_solves_succeed": all(
            (not record["proposed_stable"]) or record["support_success"]
            for record in records
        ),
        "summary_status_pass": summary_file["status"] == "PASS",
        "independent_verification": (
            verification["status"] == "PASS"
            and verification["record_count"] == 5
            and verification["tolerance"] == 1e-10
        ),
        "automated_tests": tests.returncode == 0,
        "table_exactly_matches_summary": table_text == expected_table(summaries),
        "figure_present": FIGURE.exists() and FIGURE.stat().st_size > 10000,
        "main_claim_sync": all(fragment in main_text for fragment in (
            f"{pm5['full_pass_count']}/{pm5['sample_count']} at $\\pm5\\%$",
            f"{pm10['full_pass_count']}/{pm10['sample_count']} at $\\pm10\\%$",
            "2026092705", "2026092710",
        )),
        "supplement_claim_sync": all(fragment in supp_text for fragment in (
            f"For the $\\pm5\\%$ bank, {pm5['proposed_stable_count']} proposed loops "
            f"and {pm5['baseline_stable_count']} nominal-coefficient LQR1 loops are stable, "
            f"{pm5['feasible_count']} records satisfy every non-LTR limit, "
            f"{pm5['rear_pass_count']} pass the rear comparison and "
            f"{pm5['full_pass_count']} pass all gates.",
            f"For the $\\pm10\\%$ bank, these counts are "
            f"{pm10['proposed_stable_count']}, {pm10['baseline_stable_count']}, "
            f"{pm10['feasible_count']}, {pm10['rear_pass_count']} and "
            f"{pm10['full_pass_count']}.",
        )),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "date": "2026-09-27",
        "status": status,
        "checks": checks,
        "test_stdout": tests.stdout,
        "test_stderr": tests.stderr,
        "result_sha256": sha256(ROUT / "results.json"),
        "summary_sha256": sha256(ROUT / "results_summary.json"),
        "table_sha256": sha256(TABLE),
        "figure_sha256": sha256(FIGURE),
    }
    (OUT / "release_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        f"Combined sample records & {len(records)}/1536\\\\",
        f"Finite/error-free assemblies & {sum(r['finite'] and r['error'] is None for r in records)}/1536\\\\",
        f"Independent critical records & {verification['record_count']}/5 pass\\\\",
        f"Automated robustness tests & {'4/4 pass' if tests.returncode == 0 else 'FAIL'}\\\\",
        f"Solver-derived table/figure & {'Exact match' if checks['table_exactly_matches_summary'] and checks['figure_present'] else 'FAIL'}\\\\",
        f"Release-gate status & {status}\\\\",
        r"\bottomrule",
    ]
    (OUT / "release_audit_summary.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": status, "checks": checks}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
