import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("b_lite_pause", Path(__file__).resolve().parents[1] / "scripts/watch_b_lite_qwen_pause.py")
pause = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pause)


class BLitePauseTests(unittest.TestCase):
    def plan(self):
        return {"jobs": [
            {"id": "q-dev", "model": "qwen35_9b", "stage": "qualification", "episodes": [{"key": "d"}]},
            {"id": "q-test", "model": "qwen35_9b", "stage": "test", "episodes": [{"key": "a"}, {"key": "b"}]},
            {"id": "l-test", "model": "llama31_8b", "stage": "test", "episodes": [{"key": "a"}]},
        ]}

    def result(self, root, name, state):
        folder = root / "jobs" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "result.json").write_text(json.dumps({"state": state}))

    def test_missing_rows_are_not_completion(self):
        with tempfile.TemporaryDirectory() as d:
            state, _ = pause.assess(self.plan(), Path(d))
            self.assertFalse(state["qwen_complete"])
            self.assertEqual(state["formal_expected"], 2)

    def test_qualification_does_not_inflate_formal_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.result(root, "q-dev", "complete")
            self.result(root, "q-test", "complete")
            state, done = pause.assess(self.plan(), root)
            self.assertTrue(state["qwen_complete"])
            self.assertEqual(state["audited_formal"], 2)
            self.assertEqual(len(done), 2)

    def test_quarantined_job_stays_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.result(root, "q-dev", "complete")
            self.result(root, "q-test", "quarantined")
            state, _ = pause.assess(self.plan(), root)
            self.assertFalse(state["qwen_complete"])
            self.assertEqual(state["failed_jobs"], ["q-test"])

    def test_unexpected_second_model_launch_requires_review(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "jobs/l-test/attempt1").mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                pause.assess(self.plan(), root)

    def test_pid_reuse_and_zombie_are_not_the_original_controller(self):
        wanted = {"pid": 123, "start_ticks": "1", "command": "controller", "state": "S"}
        self.assertTrue(pause.same_process(wanted, dict(wanted)))
        self.assertFalse(pause.same_process(wanted, {**wanted, "start_ticks": "2"}))
        self.assertFalse(pause.same_process(wanted, {**wanted, "state": "Z"}))
        self.assertFalse(pause.same_process(wanted, None))


if __name__ == "__main__":
    unittest.main()
