"""Consistency tests for the completed v14 combined audit."""
from __future__ import annotations

import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


class RobustnessAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((OUT / "results.json").read_text())
        cls.summary = json.loads((OUT / "results_summary.json").read_text())

    def test_complete_unique_banks(self):
        records = self.result["records"]
        self.assertEqual(len(records), 1536)
        self.assertEqual(len({record["id"] for record in records}), 1536)
        self.assertEqual(sum(r["id"].startswith("pm5-") for r in records), 512)
        self.assertEqual(sum(r["id"].startswith("pm10-") for r in records), 1024)

    def test_finite_and_solver_success(self):
        for record in self.result["records"]:
            self.assertTrue(record["finite"])
            self.assertIsNone(record["error"])
            if record["proposed_stable"]:
                self.assertTrue(record["support_success"])

    def test_counts_recompute(self):
        for bank, expected in (("pm5", 512), ("pm10", 1024)):
            records = [
                record for record in self.result["records"]
                if record["id"].startswith(bank + "-")
            ]
            summary = self.summary["summaries"][bank]
            self.assertEqual(len(records), expected)
            for field, key in (
                ("proposed_stable", "proposed_stable_count"),
                ("baseline_stable", "baseline_stable_count"),
                ("feasible", "feasible_count"),
                ("rear_pass", "rear_pass_count"),
                ("pass", "full_pass_count"),
            ):
                self.assertEqual(sum(r[field] for r in records), summary[key])

    def test_independent_verification(self):
        verification = json.loads(
            (OUT / "independent_verification.json").read_text()
        )
        self.assertEqual(verification["status"], "PASS")
        self.assertEqual(verification["record_count"], 5)
        self.assertTrue(all(record["pass"] for record in verification["records"]))


if __name__ == "__main__":
    unittest.main()