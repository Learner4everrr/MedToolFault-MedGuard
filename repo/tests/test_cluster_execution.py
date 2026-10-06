import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from medtoolbench.research.cluster_execution import AllocatedGpuLease, FileLease, install_resource_adapter
from medtoolbench.research.cluster_supervisor import next_attempt, run_command
from medtoolbench.research.runtime import GpuLease
from medtoolbench.research.supervisor import command


class ClusterExecutionTests(unittest.TestCase):
    def test_shared_job_lock_excludes_another_process(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "job.lock"
            code = ("from medtoolbench.research.cluster_execution import FileLease; import sys; "
                    "lease=FileLease(sys.argv[1]); lease.__enter__()")
            with FileLease(path):
                response = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True)
                self.assertNotEqual(response.returncode, 0)
                self.assertIn(b"BlockingIOError", response.stderr)
            self.assertEqual(subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True).returncode, 0)

    def test_inherited_lock_survives_controller_close(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "job.lock"
            lease = FileLease(path).__enter__()
            child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                                     stdin=subprocess.PIPE, pass_fds=(lease.fileno(),))
            lease.__exit__()
            try:
                with self.assertRaises(BlockingIOError):
                    with FileLease(path):
                        pass
            finally:
                child.communicate(input=b"", timeout=10)
            with FileLease(path):
                pass

    def test_resource_extension_uses_legacy_lock_on_first_two_devices(self):
        for gpu in (0, 1):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": str(gpu)}):
                from medtoolbench.research.cluster_execution import short_host
                self.assertEqual(GpuLease().path, AllocatedGpuLease({"host": short_host(), "gpu": gpu}).path)

    def test_third_and_fourth_gpu_supported_without_changing_frozen_lease(self):
        from medtoolbench.research.cluster_execution import short_host
        for gpu in (2, 3):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": str(gpu)}):
                with self.assertRaises(RuntimeError):
                    GpuLease()
                self.assertTrue(str(AllocatedGpuLease({"host": short_host(), "gpu": gpu}).path).endswith(f"gpu{gpu}.lock"))

    def test_wrong_host_and_gpu_are_rejected(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaises(RuntimeError):
                AllocatedGpuLease({"host": "not-this-host", "gpu": 0})

    def test_only_lease_binding_is_replaced(self):
        module = SimpleNamespace(GpuLease=object(), main=object(), Episode=object(),
                                 LocalModel=object(), grade=object(), recover_batches=object())
        before = dict(vars(module))
        install_resource_adapter(module, {})
        changed = [k for k, v in before.items() if getattr(module, k) is not v]
        self.assertEqual(changed, ["GpuLease"])

    def test_scientific_command_arguments_identical(self):
        plan = {"benchmark": "/b", "validation": "/v", "image": "/i",
                "conditions": ["vanilla", "medguard"], "generation": {"max_turns": 12, "seed": 42}}
        job = {"task_shards": 8, "task_shard": 3}
        system = {"path": "/m", "thinking": True, "batch_size": 2}
        original = command(plan, job, system, Path("/new"), Path("/old"))
        extended = run_command(plan, job, system, Path("/new"), Path("/old"), Path("/deploy"), "agb03-gpu3")
        self.assertEqual(original[4:], extended[8:])
        self.assertEqual(original[:3], extended[:3])

    def test_unresolved_or_exhausted_attempt_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(next_attempt(root, 2), (root / "attempt1", None))
            old = root / "attempt1"
            old.mkdir()
            with self.assertRaisesRegex(RuntimeError, "Unresolved"):
                next_attempt(root, 2)
            (old / "infrastructure_failure.txt").touch()
            (old / "config.json").write_text("{}")
            self.assertEqual(next_attempt(root, 2), (root / "attempt2", old))
            last = root / "attempt2"
            last.mkdir()
            (last / "infrastructure_failure.txt").touch()
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                next_attempt(root, 2)

    def test_completed_attempt_requires_audit_before_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / "attempt1"
            old.mkdir()
            (old / "complete.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "audited"):
                next_attempt(root, 2)


if __name__ == "__main__":
    unittest.main()
