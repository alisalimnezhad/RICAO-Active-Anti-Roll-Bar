"""Regression tests for the version-9 artifacts."""
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def test_verification_script_passes():
    subprocess.run(
        [sys.executable, str(HERE / "verify_delay_aware.py")],
        cwd=HERE.parent, check=True,
    )


def test_declared_delay_result_passes():
    data = json.loads((HERE / "outputs" / "results.json").read_text())
    assert data["summary"]["status"] == "PASS"
    assert data["summary"]["stable_count"] == 170
    assert data["summary"]["constraint_feasible_count"] == 170


def test_nonlinear_scope_and_cases_are_retained():
    data = json.loads((HERE / "outputs" / "nonlinear" / "results.json").read_text())
    assert data["summary"]["status"] == "PASS_SCREEN"
    assert data["summary"]["case_count"] == 5
    assert "not a nonlinear certificate" in data["summary"]["scope"]


def test_publication_artifacts_exist():
    required = [
        HERE.parent / "paper" / "figures" / "delay_aware_design.png",
        HERE.parent / "paper" / "figures" / "nonlinear_critical_cases.png",
        HERE.parent / "paper" / "tables" / "delay_aware_summary.tex",
        HERE.parent / "paper" / "tables" / "nonlinear_cases.tex",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in required)