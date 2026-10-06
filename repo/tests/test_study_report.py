import tempfile
import unittest
from pathlib import Path

from medtoolbench.research.analyze import cluster_gain, summarize
from medtoolbench.research.build import save_gzip
from medtoolbench.research.study_report import checked_rows


def row(task, hazard, condition, success, group="p"):
    return {"task_id": task, "hazard": hazard, "condition": condition, "group": group, "template": 1,
            "grade": {"success": success, "unsafe": []}, "action_required": False,
            "any_transient_unsafe": False, "triggered": 1, "stop": "final", "total_tokens": 10, "tool_calls": 1}


class StudyReportTests(unittest.TestCase):
    def test_patient_clustering_and_equal_task_weights(self):
        rows = [row("a", h, c, c == "medguard") for h in ("stale", "transient_503", "patient_mismatch")
                for c in ("vanilla", "medguard")]
        rows += [row("b", "stale", c, False) for c in ("vanilla", "medguard")]
        effect = cluster_gain(rows, repetitions=100)
        self.assertEqual(effect["task_weighted_gain"], .5)
        self.assertEqual(effect["patient_cluster_ci95"], [.5, .5])
        self.assertEqual(effect["groups"], 1)

    def test_conditional_eligibility_is_always_vanilla_clean(self):
        rows = [row("a", "clean", "vanilla", True), row("b", "clean", "vanilla", False),
                row("a", "clean", "medguard", False), row("b", "clean", "medguard", True)]
        rows += [row(t, "stale", c, c == "medguard") for t in ("a", "b") for c in ("vanilla", "medguard")]
        result = summarize(rows, repetitions=31)
        self.assertEqual(result["paired_guard_gain_clean_capable"]["tasks"], 1)
        self.assertEqual(result["paired_guard_gain_clean_capable"]["bootstrap_draws"], 31)
        self.assertEqual(result["conditions"]["medguard"]["fault_success_clean_capable_baseline"]["n"], 1)

    def test_missing_duplicate_and_wrong_split_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "episodes").mkdir()
            system = {"path": "/model", "thinking": False, "batch_size": 1}
            plan = {"manifest_hash": "m", "core_hashes": {}, "generation": {"seed": 1}, "conditions": ["vanilla"]}
            task = {"id": "a", "group": "p", "split": "test", "template": 1, "hazards": ["clean"]}
            job = {"task_ids": ["a"], "expected_episodes": 1}
            config = {"split": "test", "manifest_hash": "m", "core_hashes": {}, "model": "/model",
                      "thinking": False, "batch_size": 1, "seed": 1}
            with self.assertRaises(RuntimeError):
                checked_rows(root, config, plan, job, system, {"a": task})
            save_gzip(root / "episodes/a.json.gz", row("a", "clean", "vanilla", True))
            self.assertEqual(len(checked_rows(root, config, plan, job, system, {"a": task})[0]), 1)
            with self.assertRaises(RuntimeError):
                checked_rows(root, dict(config, split="dev"), plan, job, system, {"a": task})
            save_gzip(root / "episodes/duplicate.json.gz", row("a", "clean", "vanilla", True))
            with self.assertRaises(RuntimeError):
                checked_rows(root, config, plan, job, system, {"a": task})


if __name__ == "__main__":
    unittest.main()
