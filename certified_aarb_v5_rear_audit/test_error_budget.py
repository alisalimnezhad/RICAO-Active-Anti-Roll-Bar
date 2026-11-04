"""Regression checks for the archived rear-axle error-budget audit."""
from __future__ import annotations

import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "error_budget"


class ErrorBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((OUT / "error_budget.json").read_text())

    def test_status_and_archived_reconstruction(self):
        self.assertEqual(self.record["status"], "PASS")
        checks = self.record["checks"]
        self.assertTrue(checks["archived_81_phase_reconstruction"])
        self.assertTrue(checks["certification_meshes_81_and_finer_positive"])

    def test_budget_identity(self):
        for row in self.record["mesh_summary"]:
            reconstructed = (
                row["primal_sampled_gap"]
                - row["conservative_dual_residual_budget"]
                - row["intersample_remainder"]
            )
            self.assertAlmostEqual(reconstructed, row["certified_margin"], places=13)

    def test_coarse_mesh_is_reported_not_hidden(self):
        rows = {row["phases"]: row for row in self.record["mesh_summary"]}
        self.assertLess(rows[41]["certified_margin"], 0.0)
        self.assertGreater(rows[81]["certified_margin"], 0.0)
        self.assertGreater(rows[161]["certified_margin"], 0.0)
        self.assertGreater(rows[321]["certified_margin"], 0.0)
        self.assertTrue(self.record["checks"]["coarse_41_phase_bound_inconclusive"])

    def test_solver_crosscheck(self):
        for row in self.record["solver_crosscheck"]:
            self.assertTrue(row["success"])
            self.assertTrue(row["witness_feasible"])
            self.assertGreaterEqual(
                row["corrected_upper"] + 1e-12, row["primal_value"]
            )
        self.assertTrue(
            self.record["checks"]["finest_margin_dominates_solver_spread"]
        )

    def test_shipped_data_and_publication_assets_exist(self):
        for name in ("mesh_audit.csv", "solver_crosscheck.csv"):
            self.assertTrue((OUT / name).is_file(), name)
        assets = (
            HERE.parent / "paper" / "tables" / "rear_error_budget_table.tex",
            HERE.parent / "paper" / "figures" / "rear_certificate.png",
        )
        for path in assets:
            self.assertTrue(path.is_file() and path.stat().st_size > 0, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)