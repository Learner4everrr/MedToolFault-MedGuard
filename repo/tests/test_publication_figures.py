"""Small tests for the additive analysis layer; frozen runners are untouched."""
import json
import tempfile
import unittest
from pathlib import Path

from medtoolbench.publication.evidence import Evidence, digest, effect_rows
from medtoolbench.publication.diagnostics import observable, summarize


class PublicationAnalysisTests(unittest.TestCase):
    def test_stable_digest(self):
        self.assertEqual(digest({"b": 2, "a": 1}), digest({"a": 1, "b": 2}))

    def test_source_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.json"
            p.write_text('{"a":1}')
            e = Evidence(tmp)
            e.read(p)
            p.write_text('{"a":2}')
            with self.assertRaises(ValueError):
                e.unchanged()

    def test_repeated_audit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "run"
            p.mkdir()
            (p / "audit.json").write_text(json.dumps({"all_pass": True}))
            e = Evidence(tmp)
            e.audit(p)
            with self.assertRaises(ValueError):
                e.audit(p)

    def test_effects_keep_patient_weighting_and_zero_crossing(self):
        e = {"difference": .02, "ci95": [-.03, .07], "patients": 45}
        data = {"primary": {"models": {}}, "a": {"reports": {}},
                "c": {"reports": {}}, "d": {"reports": {}},
                "b": {"example": {"comparisons": {"medguard-informed": e}}}}
        row = effect_rows(data)[0]
        self.assertEqual(row["difference"], .02)
        self.assertEqual(row["ci_lower"], -.03)
        self.assertEqual(row["weighting"], "patient_equal")
        self.assertEqual(row["endpoint"], "answer_correctness")

    def test_no_write_failure_not_automatically_unsafe(self):
        row = {"grade": {"success": False, "required_matches": [False], "answer_ok": True,
                         "state_ok": False, "unsafe": []},
               "stop": "final", "generations": [], "messages": [], "trace": [],
               "answer": {"completed": True}, "action_required": True,
               "condition": "medguard", "hazard": "clean", "task_id": "t", "key": "k", "template": 1}
        got = observable(row)
        self.assertIn("claimed_completed_with_missing_write", got["tags"])
        self.assertNotIn("unsafe", got["tags"])

    def test_incomplete_diagnostic_pair_rejected(self):
        row = {"success": True, "tags": [], "stop": "final", "condition": "vanilla",
               "hazard": "clean", "task": "t", "action_required": False, "template": 1, "key": "k"}
        with self.assertRaises(ValueError):
            summarize([row])


if __name__ == "__main__":
    unittest.main()
