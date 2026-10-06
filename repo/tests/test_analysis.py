import unittest
from medtoolbench.analyze import summarize, bootstrap


def row(task, hazard, success, unsafe=False, triggered=0):
    return {"task_id": task, "hazard": hazard, "condition": "vanilla", "grade": {
        "success": success, "unsafe": ["duplicate"] if unsafe else []}, "triggered": triggered,
        "action_task": True, "tool_calls": 1, "total_tokens": 100}


class AnalysisTests(unittest.TestCase):
    def test_conditional_denominator_and_pair_weight(self):
        rows = [row("a", "clean", True), row("b", "clean", False),
                row("a", "stale", False, True, 1), row("b", "stale", True, False, 1),
                row("a", "commit_timeout", False, True, 1)]
        s = summarize(rows)["vanilla"]
        self.assertEqual(s["by_hazard"]["stale"]["conditional_robustness_n"], 1)
        self.assertEqual(s["by_hazard"]["stale"]["conditional_robustness"], 0)
        # a has two variants, b only one: underlying tasks must get equal weight.
        self.assertEqual(s["task_weighted_paired_gap"], 0)

    def test_empty_conditionals_are_not_zero_success(self):
        s = summarize([row("a", "clean", False), row("a", "stale", False)])["vanilla"]
        self.assertIsNone(s["by_hazard"]["stale"]["conditional_robustness"])
        self.assertIsNone(s["by_hazard"]["stale"]["recovery_success_rate"])

    def test_constant_cluster_bootstrap(self):
        self.assertEqual(bootstrap([1, 1, 1]), [1, 1])
