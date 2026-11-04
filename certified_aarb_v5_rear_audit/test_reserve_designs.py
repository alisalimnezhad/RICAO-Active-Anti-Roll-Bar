"""Regression checks for the final rear-reserve comparison package."""
from __future__ import annotations

import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "comparison"


class ReserveDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads((OUT / "final_reserve_audit.json").read_text())
        cls.comparison = json.loads((OUT / "reserve_comparison.json").read_text())

    def test_all_designs_pass(self):
        self.assertEqual(self.audit["status"], "PASS")
        self.assertEqual(len(self.audit["designs"]), 3)
        for record in self.audit["designs"]:
            self.assertEqual(record["status"], "PASS")
            self.assertTrue(all(record["checks"].values()))

    def test_five_percent_selection(self):
        selection = self.comparison["selection"]
        self.assertEqual(selection["key"], "reserve_5pct")
        self.assertGreaterEqual(selection["achieved_minimum_margin_percent"], 5.0)
        self.assertLess(selection["front_cost_vs_current_percent"], 0.5)

    def test_publication_artifacts_exist(self):
        names = (
            "reserve_comparison.json",
            "reserve_comparison.csv",
            "selected_controller.json",
            "selected_reserve_5pct_coefficients.csv",
            "final_reserve_audit.json",
        )
        for name in names:
            self.assertTrue((OUT / name).is_file(), name)

        assets = (
            HERE.parent / "paper" / "figures" / "rear_reserve_tradeoff.png",
            HERE.parent / "paper" / "tables" / "table_rear_reserve_tradeoff.tex",
        )
        for path in assets:
            self.assertTrue(path.is_file() and path.stat().st_size > 0, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)