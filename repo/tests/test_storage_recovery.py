import errno
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from medtoolbench.supplement.storage_recovery import Journal, atomic_json, write_probe, candidates, execute


class StorageRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.remote, self.local = self.root / "remote", self.root / "local"
        self.journal = Journal(self.remote, self.local)

    def tearDown(self):
        self.temp.cleanup()

    def test_atomic_publication_and_readback(self):
        path = self.remote / "record.json"
        atomic_json(path, {"complete": True})
        self.assertEqual(json.loads(path.read_text()), {"complete": True})
        self.assertEqual(list(path.parent.glob("*.tmp")), [])
        self.assertTrue(write_probe(self.remote / "probes")["all_pass"])
        self.assertEqual(list((self.remote / "probes").iterdir()), [])

    def test_failed_publish_preserves_prior_checkpoint(self):
        path = self.remote / "record.json"
        atomic_json(path, {"old": True})
        with patch("medtoolbench.supplement.storage_recovery.os.replace", side_effect=OSError(errno.EDQUOT, "Disk quota exceeded")):
            with self.assertRaises(OSError):
                atomic_json(path, {"new": True})
        self.assertEqual(json.loads(path.read_text()), {"old": True})
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_quota_during_heartbeat_keeps_local_journal_and_pauses_dispatch(self):
        path = self.remote / "worker.json"
        original = atomic_json
        def quota(destination, value):
            if Path(destination) == path:
                raise OSError(errno.EDQUOT, "Disk quota exceeded")
            original(destination, value)
        with patch("medtoolbench.supplement.storage_recovery.atomic_json", side_effect=quota):
            self.assertFalse(self.journal.put(path, {"pid": 42}))
        self.assertTrue(self.journal.blocked.is_set())
        alert = json.loads((self.local / "storage_alert.json").read_text())
        self.assertEqual(alert["errno"], errno.EDQUOT)
        records = [json.loads(p.read_text()) for p in self.local.glob("*.json") if p.name != "storage_alert.json"]
        self.assertEqual(records[0]["value"]["pid"], 42)
        self.assertTrue(self.journal.healthy())
        self.assertFalse(self.journal.blocked.is_set())

    def test_permission_errors_are_not_hidden_as_quota(self):
        with patch("medtoolbench.supplement.storage_recovery.atomic_json", side_effect=PermissionError(errno.EACCES, "denied")):
            with self.assertRaises(PermissionError):
                self.journal.put(self.remote / "x.json", {})

    def test_failed_health_probe_keeps_dispatch_paused(self):
        with patch("medtoolbench.supplement.storage_recovery.write_probe", side_effect=OSError(errno.ENOSPC, "No space")):
            self.assertFalse(self.journal.healthy())
        self.assertTrue(self.journal.blocked.is_set())

    def test_remote_heartbeat_failure_does_not_terminate_running_model(self):
        log = self.root / "child.log"
        self.journal.put = Mock(return_value=False)
        process = Mock()
        process.poll.side_effect = [None, 0]
        process.returncode = 0
        process.pid = 42
        with patch("medtoolbench.supplement.storage_recovery.subprocess.Popen", return_value=process), \
             patch("medtoolbench.supplement.storage_recovery.time.sleep"), \
             patch("medtoolbench.supplement.storage_recovery.stop_owned") as stop:
            result = execute(["python", "--out", str(self.root / "attempt")], log,
                {"name": "gpu0", "gpu": 0}, self.remote, [], self.journal)
        self.assertEqual(result["returncode"], 0)
        # Only the normal finally cleanup, not an early stop on put(False).
        self.assertEqual(stop.call_count, 1)
        self.assertEqual(process.poll.call_count, 2)

    def test_original_authorization_does_not_hide_interrupted_job(self):
        cp = self.remote / "plan.json"
        job = {"id": "j", "model": "m", "stage": "test"}
        folder = cp.parent / "jobs/j"
        atomic_json(folder / "isolated_execution_authorization.json", {"old": True})
        snapshot = {"interrupted": [{"campaign": str(cp), "job": "j"}]}
        with patch("medtoolbench.supplement.storage_recovery.model_gate", return_value="passed"):
            ready = candidates([(cp, {"jobs": [job]})], snapshot, set(), {})
        self.assertEqual(len(ready), 1)
        self.assertFalse(ready[0][0])

    def test_completed_and_active_jobs_are_not_restarted(self):
        cp = self.remote / "plan.json"
        job = {"id": "j", "model": "m", "stage": "test"}
        with patch("medtoolbench.supplement.storage_recovery.model_gate", return_value="passed"):
            self.assertEqual(candidates([(cp, {"jobs": [job]})], {"interrupted": []}, {(str(cp), "j")}, {}), [])
            atomic_json(cp.parent / "jobs/j/result.json", {"state": "complete"})
            self.assertEqual(candidates([(cp, {"jobs": [job]})], {"interrupted": []}, set(), {}), [])

    def test_failed_model_and_qualification_gate_still_block_test(self):
        cp = self.remote / "plan.json"
        job = {"id": "j", "model": "m", "stage": "test"}
        plans = [(cp, {"jobs": [job]})]
        self.assertEqual(candidates(plans, {"interrupted": []}, set(), {"m": "oom"}), [])
        with patch("medtoolbench.supplement.storage_recovery.model_gate", return_value="waiting"):
            self.assertEqual(candidates(plans, {"interrupted": []}, set(), {}), [])


if __name__ == "__main__":
    unittest.main()
