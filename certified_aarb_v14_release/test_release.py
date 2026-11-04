"""Tests for the final v14 release gate."""
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent


class ReleaseTests(unittest.TestCase):
    def test_release_audit_passes(self):
        audit = json.loads((HERE / "outputs" / "release_audit.json").read_text())
        self.assertEqual(audit["status"], "PASS")
        self.assertTrue(all(audit["checks"].values()))

    def test_summary_reports_pass(self):
        text = (HERE / "outputs" / "release_audit_summary.tex").read_text()
        self.assertIn("Release-gate status & PASS", text)


if __name__ == "__main__":
    unittest.main()