import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.research.build import save_gzip
from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.report import generate, paired


class SupplementReportTests(unittest.TestCase):
    def test_hazard_and_branch_replicas_do_not_multiply_parent_weight(self):
        rows = []
        for task, parent, hazards, gain in (("a", "p1", 4, 1), ("a2", "p1", 2, 1), ("b", "p2", 1, 0)):
            for h in range(hazards):
                for condition, success in (("vanilla", 0), ("medguard", gain)):
                    rows.append({"task": task, "parent": parent, "group": parent, "hazard": str(h), "condition": condition, "success": success})
        result = paired(rows, "medguard", draws=100)
        self.assertEqual(result["difference"], .5)
        self.assertEqual(result["parents"], 2)
        self.assertEqual(result["groups"], 2)

    def test_pooled_seeds_are_averaged_before_patient_resampling(self):
        rows = [{"task": "a", "parent": "p1", "group": "p1", "hazard": "fault", "condition": c, "success": s}
            for c, s in (("vanilla", 0), ("vanilla", 1), ("medguard", 1), ("medguard", 1))]
        self.assertEqual(paired(rows, "medguard", draws=100)["difference"], .5)

    def fixture(self, root, external=False):
        # Deliberately synthetic records: exercise report plumbing, never research data.
        plan = {"plan_hash": "synthetic-unit-test", "jobs": []}
        for stage in ("qualification", "test"):
            folder = root / "jobs" / stage
            run = folder / "attempt1"
            (run / "episodes").mkdir(parents=True)
            entries = []
            for hazard in ("clean", "fault"):
                for condition in ("vanilla", "medguard"):
                    key = hazard + "--" + condition
                    row = {"key": key, "task_id": "synthetic", "group": "patient", "template": 1,
                        "hazard": hazard, "condition": condition, "input_tokens": 3, "output_tokens": 2,
                        "tool_calls": 1, "raw_tool_calls": 1, "triggered": hazard != "clean", "stop": "final",
                        "action_required": not external, "grade": {"success": condition == "medguard", "unsafe": []}}
                    if external:
                        row.update(question_id="synthetic", success=condition == "medguard")
                    save_gzip(run / "episodes" / (key + ".json.gz"), row)
                    entries.append({"key": key})
            write_json(run / "audit.json", {"all_pass": True, "episodes": 4})
            write_json(folder / "result.json", {"state": "complete", "run": str(run)})
            plan["jobs"].append({"id": stage, "stage": stage, "episodes": entries, "arm": "test", "model": "mock"})
        return plan

    def test_complete_report_excludes_development(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.fixture(root)
            with patch("medtoolbench.supplement.report.load_plan", return_value=plan):
                generate(root / "plan.json")
            complete = json.loads((root / "report/complete.json").read_text())
            metrics = json.loads((root / "report/metrics.json").read_text())
            self.assertEqual(complete["episodes_test"], 4)
            self.assertEqual(metrics["reports"]["test/mock"]["comparisons"]["medguard-vanilla"]["difference"], 1)

    def test_read_only_report_does_not_claim_write_safety(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.fixture(root, external=True)
            with patch("medtoolbench.supplement.report.load_plan", return_value=plan):
                generate(root / "plan.json")
            metrics = json.loads((root / "report/metrics.json").read_text())
            for condition in metrics["reports"]["test/mock"]["conditions"].values():
                self.assertIsNone(condition["unsafe_state"])
                self.assertIsNone(condition["action_required_unsafe_state"])

    def test_failed_audit_never_produces_complete_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.fixture(root)
            write_json(root / "jobs/test/attempt1/audit.json", {"all_pass": False, "episodes": 4})
            with patch("medtoolbench.supplement.report.load_plan", return_value=plan):
                with self.assertRaises(RuntimeError):
                    generate(root / "plan.json")
            self.assertFalse((root / "report/complete.json").exists())


if __name__ == "__main__":
    unittest.main()
