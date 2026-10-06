import copy
import tempfile
import unittest
from pathlib import Path

from medtoolbench.construction import suite
from medtoolbench.grading import grade
from medtoolbench.hazards import Proxy
from medtoolbench.oracle import solve
from medtoolbench.storage import Store, digest


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.tasks = suite(8)

    def tearDown(self):
        self.temp.cleanup()

    def store(self, t):
        return Store(Path(self.temp.name) / f"{t['id']}.sqlite", t["resources"])

    def test_all_clean_and_recovery_oracles(self):
        for task in suite():
            for hazard in task["hazards"]:
                with self.subTest(task=task["id"], hazard=hazard):
                    s = self.store(task)
                    before = s.snapshot()
                    p = Proxy(s, task, hazard)
                    answer = solve(task, p)
                    g = grade(task, before, s.snapshot(), answer)
                    self.assertTrue(g["success"], g)
                    self.assertEqual(p.triggered, int(hazard != "clean"))
                    s.reset()
                    self.assertEqual(before, s.snapshot())
                    s.close()

    def test_determinism_and_isolation(self):
        task = self.tasks[0]
        traces = []
        for _ in range(2):
            s = self.store(task)
            p = Proxy(s, task, "stale")
            normal = s.call("GET", "/Patient/" + task["patient"])
            self.assertEqual(p.call("GET", "/Patient/" + task["patient"]), normal)
            self.assertEqual(p.triggered, 0)
            solve(task, p)
            traces.append(p.trace)
            s.close()
        self.assertEqual(traces[0], traces[1])

    def test_wrong_states_rejected(self):
        task = self.tasks[1]
        for corruption in ("missing", "duplicate", "wrong_patient", "wrong_dose", "collateral", "deleted", "extra_time", "extra_instruction"):
            with self.subTest(corruption=corruption):
                s = self.store(task)
                before = s.snapshot()
                action = copy.deepcopy(task["action_template"])
                if corruption == "wrong_patient":
                    action["subject"]["reference"] = "Patient/" + task["distractor"]
                elif corruption == "wrong_dose":
                    action["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]["value"] = 20
                elif corruption == "extra_time":
                    action["dosageInstruction"][0]["timing"]["repeat"]["duration"] = 20
                elif corruption == "extra_instruction":
                    action["dosageInstruction"][0]["asNeededBoolean"] = True
                if corruption != "missing":
                    s.call("POST", "/MedicationRequest", action)
                if corruption == "duplicate":
                    s.call("POST", "/MedicationRequest", action)
                if corruption == "collateral":
                    changed = copy.deepcopy(task["resources"][0])
                    changed["active"] = False
                    s.call("PUT", "/Patient/" + task["patient"], changed)
                if corruption == "deleted":
                    s.call("DELETE", "/Patient/" + task["patient"])
                g = grade(task, before, s.snapshot(), {"action": "ordered"})
                self.assertFalse(g["success"])
                if corruption != "missing":
                    self.assertTrue(g["unsafe"])
                s.close()

    def test_no_action_is_not_always_success(self):
        for task in self.tasks:
            s = self.store(task)
            g = grade(task, s.snapshot(), s.snapshot(), task["contract"]["answer"])
            self.assertEqual(g["success"], not task["contract"]["required_creations"])
            s.close()
        no_action = self.tasks[5]
        s = self.store(no_action)
        before = s.snapshot()
        s.call("POST", "/MedicationRequest", no_action["action_template"])
        self.assertIn("unnecessary_action", grade(no_action, before, s.snapshot(), {"action": "none"})["unsafe"])
        s.close()

    def test_committed_write_is_persistent_before_error(self):
        task = self.tasks[2]
        results = []
        for h in ("commit_timeout", "no_commit_timeout"):
            s = self.store(task)
            before = s.snapshot()
            p = Proxy(s, task, h)
            results.append(p.call("POST", "/Observation", task["action_template"]))
            reopened = __import__("sqlite3").connect(str(Path(self.temp.name) / f"{task['id']}.sqlite"))
            count = reopened.execute("SELECT count(*) FROM resources").fetchone()[0]
            self.assertEqual(count, len(before) + int(h == "commit_timeout"))
            reopened.close()
            s.close()
        self.assertEqual(results[0], results[1])

    def test_retry_can_create_duplicate(self):
        task = self.tasks[2]
        s = self.store(task)
        before = s.snapshot()
        p = Proxy(s, task, "commit_timeout")
        answer = solve(task, p, "retry_only")
        g = grade(task, before, s.snapshot(), answer)
        self.assertIn("duplicate", g["unsafe"])
        self.assertFalse(g["success"])
        s.close()

    def test_clean_proxy_equivalence(self):
        task = self.tasks[0]
        s = self.store(task)
        p = Proxy(s, task)
        path = "/Observation?patient=" + task["patient"]
        self.assertEqual(s.call("GET", path), p.call("GET", path))
        self.assertEqual(p.triggered, 0)
        s.close()

    def test_guard_has_no_gold_or_hazard_hint(self):
        task = self.tasks[1]
        s = self.store(task)
        before = s.snapshot()
        proxy = Proxy(s, task, "commit_timeout")
        public = {k: v for k, v in task.items() if k not in {"resources", "contract", "hazards", "distractor"}}
        answer = solve(public, proxy)
        self.assertTrue(grade(task, before, s.snapshot(), answer)["success"])
        s.close()


if __name__ == "__main__":
    unittest.main()
