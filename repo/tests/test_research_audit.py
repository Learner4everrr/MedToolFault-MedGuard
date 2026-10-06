import copy
import unittest

from medtoolbench.research.audit import apply_diff, transient_issues
from medtoolbench.research.analyze import clean_drop, cluster_gain
from medtoolbench.research.contracts import grade, service
from medtoolbench.research.trace_diagnostics import tags


class ResearchAuditTests(unittest.TestCase):
    def test_replay_rejects_wrong_preimage(self):
        before = [{"resourceType": "Patient", "id": "p", "gender": "female"}]
        wrong = dict(before[0], gender="male")
        for diff in ({"removed": [wrong], "added": [], "changed": []},
                     {"removed": [], "added": [], "changed": [{"before": wrong, "after": before[0]}]}):
            with self.assertRaises(ValueError):
                apply_diff(before, diff)

    def test_transient_unsafe_replays_even_after_final_repair(self):
        before = [{"resourceType": "Patient", "id": "p"}]
        task = {"patient": "p", "contract": {"answer": {"completed": True}, "required_creations": []}}
        state = before + [dict(service("p", "4548-4"), id="new")]
        row = {"trace": [{"method": "POST", "response": {"status": 201}, "committed_response": None},
                          {"method": "DELETE", "response": {"status": 204}, "committed_response": None}],
               "transient_states": [{"trace_index": 0, "grade": grade(task, before, state, {})},
                                    {"trace_index": 1, "grade": grade(task, before, before, {})}],
               "any_transient_unsafe": True}
        self.assertFalse(transient_issues(row, task, before))
        tampered = copy.deepcopy(row)
        tampered["transient_states"][0]["grade"]["unsafe"] = []
        self.assertIn("transient regrade mismatch: unsafe", transient_issues(tampered, task, before))
        row["transient_states"].pop(0)
        self.assertTrue(transient_issues(row, task, before))

    def test_cluster_effect_does_not_overweight_hazard_replicates(self):
        rows = []
        for task, hazards, gain in (("a", ["h1", "h2", "h3"], True), ("b", ["h1"], False)):
            for h in hazards:
                for condition in ("vanilla", "medguard"):
                    rows.append({"task_id": task, "group": task, "hazard": h, "condition": condition,
                                 "grade": {"success": condition == "medguard" and gain}})
        value = cluster_gain(rows, repetitions=100)
        self.assertEqual(value["task_weighted_gain"], .5)
        self.assertEqual(value["groups"], 2)
        self.assertEqual(value, cluster_gain(rows, repetitions=100))

    def test_clean_drop_pairs_with_same_condition(self):
        rows = [{"task_id": "a", "group": "p", "condition": c, "hazard": h, "grade": {"success": ok}}
                for c, h, ok in [("vanilla", "clean", True), ("vanilla", "fault", False),
                                 ("medguard", "clean", False), ("medguard", "fault", True)]]
        self.assertEqual(clean_drop(rows, "vanilla", repetitions=100)["task_weighted_clean_minus_fault"], 1)
        self.assertEqual(clean_drop(rows, "medguard", repetitions=100)["task_weighted_clean_minus_fault"], -1)

    def test_blind_retry_tag_requires_no_same_type_read(self):
        write = {"method": "POST", "path": "/ServiceRequest", "body": service("p", "4548-4"), "response": {"status": 504}}
        row = {"trace": [write, dict(write, response={"status": 201})], "stop": "final", "answer": {},
               "grade": {"success": False, "state_ok": False, "answer_ok": False, "required_matches": [], "unsafe": []},
               "any_transient_unsafe": False}
        name = "equivalent_post_retry_without_intervening_same_type_read"
        self.assertIn(name, tags(row))
        row["trace"].insert(1, {"method": "GET", "path": "/ServiceRequest?subject=p", "response": {"status": 200}})
        self.assertNotIn(name, tags(row))


if __name__ == "__main__":
    unittest.main()
