import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.supplement import b_lite_dev_pause as pause


class DevelopmentPauseTests(unittest.TestCase):
    def test_real_process_identity_is_stable(self):
        first = pause.identity(os.getpid())
        self.assertEqual(first["pid"], os.getpid())
        self.assertEqual(first["ppid"], os.getppid())
        self.assertTrue(pause.same_process(first, pause.identity(os.getpid())))

    def test_pid_reuse_does_not_match(self):
        self.assertFalse(pause.same_process({"pid": 2, "start_ticks": "10"}, {"pid": 2, "start_ticks": "11"}))
        self.assertFalse(pause.same_process({"pid": 2, "start_ticks": "10"}, None))

    def test_formal_directory_invalidates_zero_formal_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            control = Path(temp)
            plan = {"jobs": [{"id": "heldout", "stage": "test"}]}
            pause.no_formal(plan, control)
            (control / "jobs/heldout").mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                pause.no_formal(plan, control)

    def test_live_child_is_not_complete(self):
        process = {"pid": 2, "start_ticks": "10", "state": "R", "exit_status": None}
        with patch.object(pause, "identity", return_value=process):
            self.assertIsNone(pause.exit_receipt({"process": process, "job": "dev"}))

    def test_exit_status_preserved(self):
        process = {"pid": 2, "start_ticks": "10", "state": "Z", "exit_status": 256}
        with patch.object(pause, "identity", return_value=process):
            self.assertEqual(pause.exit_receipt({"process": process, "job": "dev"})["returncode"], 1)

    def test_dispatcher_cannot_be_killed_while_child_live(self):
        with patch.object(pause, "no_formal"), patch.object(pause.os, "kill") as kill:
            with self.assertRaises(RuntimeError):
                pause.finish(Path("unused"), Path("unused"), {"children": [{}]}, {}, [None])
            kill.assert_not_called()

    def test_child_is_rechecked_before_dispatcher_signal(self):
        with patch.object(pause, "no_formal"), patch.object(pause, "exit_receipt", return_value=None), \
                patch.object(pause.os, "kill") as kill:
            with self.assertRaises(RuntimeError):
                pause.finish(Path("unused"), Path("unused"), {"children": [{}]}, {}, [{}])
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
