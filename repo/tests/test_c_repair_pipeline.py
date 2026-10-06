import copy
import json
import tempfile
import unittest
from pathlib import Path

from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.c_repair_pipeline import finalize, predecessor_ready
from medtoolbench.supplement.profile_plan import experiment_identity, validate_repair


class CRepairPipelineTests(unittest.TestCase):
    def fixture_plans(self):
        parent = {"plan_hash": "old", "source_hashes": {"old.py": "unchanged"},
            "runner_module": "old_runner", "tasks": [{"id": "t", "answer": 5}], "workers": [{"gpu": 0}],
            "jobs": [{"settings": {"seed": 42, "raw_call_budget": 48}}], "primary_core_hashes": {"grader": "fixed"}}
        revised = copy.deepcopy(parent)
        revised.update(plan_hash="new", runner_module="medtoolbench.supplement.profile_runner",
            profile_correction={"parent_plan_hash": "old", "experiment_identity": experiment_identity(parent)})
        revised["source_hashes"]["new.py"] = "new-implementation"
        return parent, revised

    def test_repair_allows_additive_adapter_only(self):
        parent, revised = self.fixture_plans()
        self.assertTrue(validate_repair(parent, revised))
        revised["source_hashes"]["old.py"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            validate_repair(parent, revised)

    def test_repair_cannot_change_labels_or_budgets(self):
        for category in ("label", "budget"):
            parent, revised = self.fixture_plans()
            if category == "label":
                revised["tasks"][0]["answer"] = 6
            else:
                revised["jobs"][0]["settings"]["raw_call_budget"] = 96
            with self.assertRaisesRegex(RuntimeError, "design"):
                validate_repair(parent, revised)

    def test_predecessor_must_be_terminal_before_gpu_handoff(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertFalse(predecessor_ready(root, "gpu0"))
            for state in ("executing_campaign", "waiting_for_campaign", "reporting"):
                write_json(root / "gpu0.json", {"state": state})
                self.assertFalse(predecessor_ready(root, "gpu0"))
            for state in ("complete", "completed_with_failures"):
                write_json(root / "gpu0.json", {"state": state})
                self.assertTrue(predecessor_ready(root, "gpu0"))

    def test_interrupted_predecessor_requires_explicit_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "gpu0.json", {"state": "interrupted_or_failed"})
            with self.assertRaisesRegex(RuntimeError, "recovery"):
                predecessor_ready(root, "gpu0")

    def test_combined_report_needs_all_canonical_campaigns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            campaigns = [root / label / "plan.json" for label in ("a", "b", "c4", "d")]
            config = {"campaigns": [str(p) for p in campaigns], "plan_hashes": ["p"] * 4,
                "episodes": 8, "pipeline_hash": "pipeline"}
            for cp in campaigns:
                (cp.parent / "report").mkdir(parents=True)
                write_json(cp.parent / "complete.json", {"all_pass": True, "plan_hash": "p"})
                write_json(cp.parent / "report/metrics.json", {"unit_test": True})
                if cp.parent.name != "c4":
                    write_json(cp.parent / "report/complete.json", {"all_pass": True, "plan_hash": "p", "episodes_test": 1})
            pipeline = root / "pipeline.json"
            finalize(pipeline, config)
            self.assertFalse((root / "complete.json").exists())
            write_json(campaigns[2].parent / "report/complete.json", {"all_pass": True, "plan_hash": "p", "episodes_test": 1})
            finalize(pipeline, config)
            completion = json.loads((root / "complete.json").read_text())
            self.assertTrue(completion["all_pass"])
            self.assertEqual(completion["episodes_test"], 4)


if __name__ == "__main__":
    unittest.main()
