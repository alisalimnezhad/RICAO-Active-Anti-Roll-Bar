"""Regression tests for the version-10 presentation artifacts."""
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def test_verification_passes():
    subprocess.run([sys.executable, str(HERE / "verify_presentation.py")],
                   cwd=HERE.parent, check=True)


def test_required_publication_artifacts_exist():
    paths = [
        HERE.parent / "paper/figures/certification_architecture.png",
        HERE.parent / "paper/tables/certificate_scope.tex",
        HERE.parent / "paper/tables/complexity_implementation.tex",
        HERE.parent / "paper/tables/complexity_performance.tex",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_scope_keeps_nonclaims_explicit():
    data = json.loads((HERE / "outputs/presentation_data.json").read_text())
    excluded = " ".join(row["excluded"] for row in data["scope_rows"])
    assert "Interval proof" in excluded
    assert "Experiment" in excluded
    assert "Observer" in excluded


def test_complexity_inventory_is_monotone_where_expected():
    data = json.loads((HERE / "outputs/presentation_data.json").read_text())
    rows = {row["controller"]: row for row in data["controllers"]}
    assert rows["Sampled LQR1"]["stored"] < rows["Rear protected"]["stored"]
    assert rows["Rear protected"]["stored"] < rows["Scheduled balanced"]["stored"]