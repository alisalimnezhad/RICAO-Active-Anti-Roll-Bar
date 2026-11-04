"""Regression tests for the fixed-controller sensitivity package."""
from __future__ import annotations

import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


class SensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads((OUT / "results.json").read_text())
        cls.audit = json.loads((OUT / "verification.json").read_text())

    def test_audit_passes(self):
        self.assertEqual(self.audit["status"], "PASS")
        self.assertTrue(all(self.audit["checks"].values()))

    def test_declared_case_counts(self):
        self.assertEqual(len(self.results["cases"]), 48)
        self.assertEqual(len(self.results["delay"]), 4)
        self.assertTrue(all(not case["redesign_performed"] for case in self.results["cases"]))

    def test_expected_failures_are_not_hidden(self):
        statuses = {case["key"]: case["status"] for case in self.results["cases"]}
        self.assertNotEqual(statuses["fir_1"], "PASS")
        self.assertNotEqual(statuses["fir_2"], "PASS")
        self.assertEqual(statuses["fir_4"], "PASS")
        self.assertEqual(statuses["fir_6"], "PASS")
        self.assertTrue(any(value == "FAIL_UNSTABLE" for value in statuses.values()))

    def test_artifacts_exist(self):
        assets = (
            HERE.parent / "paper" / "figures" / "fixed_sensitivity_summary.png",
            HERE.parent / "paper" / "figures" / "delay_hydraulic_summary.png",
            HERE.parent / "paper" / "tables" / "sensitivity_fir_horizon.tex",
            HERE.parent / "paper" / "tables" / "sensitivity_delay.tex",
            HERE.parent / "paper" / "tables" / "sensitivity_hydraulic.tex",
        )
        for path in assets:
            self.assertTrue(path.is_file() and path.stat().st_size > 0, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)