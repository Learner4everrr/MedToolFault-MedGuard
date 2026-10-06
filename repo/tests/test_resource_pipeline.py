import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.research.cluster_execution import AllocatedGpuLease, short_host
from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.campaign import gate
from medtoolbench.supplement.resource_pipeline import load, rebind, scientific_identity, terminal


class ResourcePipelineTests(unittest.TestCase):
    def setUp(self):
        self.workers = [{"name": "node-gpu" + str(i), "host": "node", "gpu": i, "uuid": "GPU-" + str(i), "slurm_job": 1} for i in range(8)]
        self.parent = {"plan_hash": "parent-hash", "workers": [{"name": "old"}], "created_at": 1, "session": "old",
            "jobs": [{"id": "j", "stage": "test", "episodes": [{"key": "k"}], "settings": {"seed": 42, "max_turns": 12}}],
            "systems": [{"batch_size": 2, "path": "/local", "thinking": False}], "tasks": [{"id": "t", "gold": "x"}],
            "source_hashes": {"grader": "fixed"}, "input_hashes": {"data": "fixed"}, "require_oracle": True}

    def test_resource_rebind_preserves_all_scientific_fields(self):
        before = copy.deepcopy(self.parent)
        revised = rebind(self.parent, Path("/parent/plan.json"), self.workers, "new")
        self.assertEqual(scientific_identity(revised), scientific_identity(self.parent))
        self.assertEqual(self.parent, before)
        self.assertEqual(revised["jobs"], before["jobs"])
        self.assertEqual(revised["systems"], before["systems"])
        self.assertEqual(revised["source_hashes"], before["source_hashes"])
        self.assertEqual(revised["workers"], self.workers)
        self.assertNotEqual(revised["plan_hash"], before["plan_hash"])

    def test_changed_seed_is_not_a_resource_change(self):
        revised = rebind(self.parent, Path("/parent/plan.json"), self.workers, "new")
        revised["jobs"][0]["settings"]["seed"] = 43
        self.assertNotEqual(scientific_identity(revised), scientific_identity(self.parent))

    def test_load_checks_original_scientific_identity(self):
        revised = rebind(self.parent, Path("/parent/plan.json"), self.workers, "new")
        config = {"campaigns": ["/new/plan.json"], "worker_count": 8}
        def read(path):
            return self.parent if str(path) == "/parent/plan.json" else revised
        with tempfile.TemporaryDirectory() as temporary:
            with patch("medtoolbench.supplement.resource_pipeline.load_base", return_value=config), patch("medtoolbench.supplement.resource_pipeline.load_plan", side_effect=read):
                self.assertEqual(load(Path(temporary) / "pipeline.json")[0], config)
                revised["systems"][0]["batch_size"] = 3
                with self.assertRaisesRegex(RuntimeError, "Scientific"):
                    load(Path(temporary) / "pipeline.json")

    def test_duplicate_physical_gpu_is_rejected(self):
        revised = rebind(self.parent, Path("/parent/plan.json"), self.workers, "new")
        revised["workers"][7]["uuid"] = revised["workers"][6]["uuid"]
        with patch("medtoolbench.supplement.resource_pipeline.load_base", return_value={"campaigns": ["/new"], "worker_count": 8}), patch("medtoolbench.supplement.resource_pipeline.load_plan", return_value=revised):
            with self.assertRaisesRegex(RuntimeError, "unique"):
                load(Path("/missing-pipeline.json"))

    def test_all_eight_workers_required_for_qualification_failure(self):
        names = [w["name"] for w in self.workers]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "workers").mkdir()
            for name in names[:6]:
                write_json(root / "workers" / (name + ".json"), {"state": "qualification_failed"})
            self.assertIsNone(terminal(root, names))
            for name in names[6:]:
                write_json(root / "workers" / (name + ".json"), {"state": "qualification_failed"})
            self.assertIs(terminal(root, names), False)

    def test_oracle_gate_still_prevents_test_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "oracle").mkdir()
            plan = {"plan_hash": "expected", "require_oracle": True, "jobs": []}
            self.assertEqual(gate(root, plan), "waiting")
            write_json(root / "oracle/complete.json", {"all_pass": True, "plan_hash": "wrong"})
            self.assertEqual(gate(root, plan), "failed")
            write_json(root / "oracle/complete.json", {"all_pass": True, "plan_hash": "expected"})
            self.assertEqual(gate(root, plan), "passed")

    def test_gpu_four_through_seven_keep_exclusive_lease(self):
        # Unit tests must not contend with genuine experimental GPU leases.
        with tempfile.TemporaryDirectory() as temporary:
            for gpu in range(4, 8):
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": str(gpu)}):
                    row = {"host": short_host(), "gpu": gpu}
                    first, second = AllocatedGpuLease(row), AllocatedGpuLease(row)
                    self.assertTrue(first.path.name.endswith(f"gpu{gpu}.lock"))
                    first.path = second.path = Path(temporary) / first.path.name
                    with first:
                        with self.assertRaises(BlockingIOError):
                            with second:
                                pass


if __name__ == "__main__":
    unittest.main()
