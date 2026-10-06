import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.supplement.memory_execution import authorize_oom_repair, verify_completion


class RecoveryEligibility(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.job = {"model": "gemma3_12b"}
        (self.root / "attempt1").mkdir()
        self.write(self.root / "result.json", {"state": "failed"})
        self.write(self.root / "attempt1/failure.json", {"error": "torch.OutOfMemoryError: CUDA out of memory"})

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, p, value):
        p.write_text(json.dumps(value))

    def test_oom_allowed_with_record(self):
        proof = authorize_oom_repair(self.root, self.job)
        self.assertFalse(proof["prior_partial_rows_pooled"])
        self.assertEqual(len(proof["prior_failures"]), 1)

    def test_accuracy_never_used(self):
        self.write(self.root / "result.json", {"state": "complete", "success_rate": 0})
        with self.assertRaises(RuntimeError):
            authorize_oom_repair(self.root, self.job)

    def test_audit_rejects_retry(self):
        self.write(self.root / "attempt1/audit.json", {"all_pass": False})
        with self.assertRaises(RuntimeError):
            authorize_oom_repair(self.root, self.job)

    def test_other_failure_rejected(self):
        self.write(self.root / "attempt1/failure.json", {"error": "HAPI reset mismatch"})
        with self.assertRaises(RuntimeError):
            authorize_oom_repair(self.root, self.job)

    def test_missing_failure_rejected(self):
        (self.root / "attempt2").mkdir()
        with self.assertRaises(RuntimeError):
            authorize_oom_repair(self.root, self.job)

    def test_other_model_rejected(self):
        with self.assertRaises(RuntimeError):
            authorize_oom_repair(self.root, {"model": "other"})

    def test_mixed_failure_history_is_not_classified_as_oom(self):
        (self.root / "attempt2").mkdir()
        self.write(self.root / "attempt2/failure.json", {"error": "Timeout while importing fixture"})
        with self.assertRaises(RuntimeError):
            authorize_oom_repair(self.root, self.job)

    def test_completed_attempt_never_retried_even_if_result_says_failed(self):
        self.write(self.root / "attempt1/complete.json", {"all_pass": True})
        with self.assertRaises(RuntimeError):
            authorize_oom_repair(self.root, self.job)

    def test_full_identity_required_for_completion(self):
        job = {"episodes": [{"key": "a"}]}
        self.write(self.root / "complete.json", {"all_pass": True})
        self.write(self.root / "audit.json", {"all_pass": True, "episodes": 1, "plan_hash": "p"})
        self.write(self.root / "config.json", {"plan_hash": "p", "job": job})
        self.assertTrue(verify_completion(self.root, {"plan_hash": "p"}, job)["all_pass"])
        with self.assertRaises(RuntimeError):
            verify_completion(self.root, {"plan_hash": "wrong"}, job)

    def test_missing_keys_reject_completion(self):
        job = {"episodes": [{"key": "a"}, {"key": "b"}]}
        self.write(self.root / "complete.json", {"all_pass": True})
        self.write(self.root / "audit.json", {"all_pass": True, "episodes": 2, "plan_hash": "p", "missing": ["a"]})
        self.write(self.root / "config.json", {"plan_hash": "p", "job": job})
        with self.assertRaises(RuntimeError):
            verify_completion(self.root, {"plan_hash": "p"}, job)


class FeedForwardTiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def model(self, dtype=None):
        import torch
        from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
        from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP
        torch.manual_seed(4)
        return Gemma3MLP(Gemma3TextConfig(hidden_size=16, intermediate_size=32)).to(dtype=dtype or torch.float32).eval()

    def test_all_tokens_batches_and_remainder_preserved(self):
        import torch
        from medtoolbench.supplement.memory_adapter import install, uninstall
        for dtype in (torch.float32, torch.bfloat16):
            for length in (1, 16, 17, 35):
                model = self.model(dtype)
                x = torch.randn(2, length, 16, dtype=dtype)
                untouched = x.clone()
                with torch.inference_mode():
                    baseline = model(x)
                    record = install(model, 16)
                    fixed = model(x)
                    torch.testing.assert_close(fixed, baseline)
                    torch.testing.assert_close(x, untouched)
                    self.assertFalse(record["attention_changed"])
                    uninstall(model)
                    torch.testing.assert_close(model(x), baseline)

    def test_training_and_gradients_refused(self):
        import torch
        from medtoolbench.supplement.memory_adapter import install
        model = self.model()
        install(model, 8)
        with self.assertRaises(RuntimeError):
            model(torch.zeros(2, 16, 16))
        model.train()
        with torch.inference_mode(), self.assertRaises(RuntimeError):
            model(torch.zeros(2, 16, 16))

    def test_duplicate_and_invalid_installation_refused(self):
        from medtoolbench.supplement.memory_adapter import install
        model = self.model()
        with self.assertRaises(ValueError):
            install(model, 0)
        install(model)
        with self.assertRaises(RuntimeError):
            install(model)

    def test_tiling_does_not_consume_sampling_rng(self):
        import torch
        from medtoolbench.supplement.memory_adapter import install
        model = self.model()
        install(model, 8)
        x = torch.randn(2, 35, 16)
        state = torch.get_rng_state().clone()
        with torch.inference_mode():
            model(x)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


class ExecutionManifest(unittest.TestCase):
    def test_canonical_identity_and_source_hash_are_enforced(self):
        from medtoolbench.storage import digest
        from medtoolbench.supplement.memory_execution import load
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "adapter.py"
            source.write_text("implementation")
            base = {"pipeline_hash": "base", "campaigns": ["a"], "plan_hashes": ["p"]}
            manifest = {"canonical_pipeline": "canonical.json", "canonical_pipeline_hash": "base",
                        "campaigns": ["a"], "plan_hashes": ["p"], "source_hashes": {str(source): digest("implementation")},
                        "chunk_tokens": 512, "extra_infrastructure_attempts": 2}
            path = root / "manifest.json"
            def save():
                path.write_text(json.dumps({**manifest, "manifest_hash": digest(manifest)}))
            save()
            with patch("medtoolbench.supplement.memory_execution.load_pipeline", return_value=base):
                self.assertEqual(load(path)["plan_hashes"], ["p"])
                manifest["plan_hashes"] = ["changed-budget"]
                save()
                with self.assertRaises(RuntimeError):
                    load(path)
                manifest["plan_hashes"] = ["p"]
                save()
                source.write_text("modified")
                with self.assertRaises(RuntimeError):
                    load(path)


class ControllerBoundaryIntegration(unittest.TestCase):
    def test_live_job_is_not_restarted_and_shadow_gate_precedes_repair(self):
        from medtoolbench.research.cluster_execution import FileLease
        from medtoolbench.research.job_state import write_json
        from medtoolbench.supplement import memory_execution as execution
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cp, control = root / "a/plan.json", root / "memory"
            control.mkdir()
            (control / "preflight").mkdir()
            (cp.parent / "workers").mkdir(parents=True)
            row = {"name": "test-gpu0", "host": "test", "gpu": 0}
            jobs = [{"id": "dev", "model": "gemma3_12b", "stage": "qualification", "episodes": [{"key": "d"}]},
                    {"id": "active", "model": "gemma3_12b", "stage": "test", "episodes": [{"key": "a"}]},
                    {"id": "oom", "model": "gemma3_12b", "stage": "test", "episodes": [{"key": "o"}]}]
            plan = {"workers": [row], "jobs": jobs, "plan_hash": "plan", "python": "unused"}
            manifest = {"manifest_hash": "manifest", "campaigns": [str(cp)], "execution_order": [0]}
            for job in jobs:
                (cp.parent / "jobs" / job["id"]).mkdir(parents=True)
            write_json(cp.parent / "jobs/dev/result.json", {"state": "complete", "episodes": 1})
            failed = cp.parent / "jobs/oom"
            (failed / "attempt1").mkdir()
            write_json(failed / "result.json", {"state": "failed"})
            write_json(failed / "attempt1/failure.json", {"error": "torch.OutOfMemoryError: CUDA out of memory"})
            write_json(control / "preflight/complete.json", {"all_pass": True, "manifest_hash": "manifest"})
            old_status = cp.parent / "workers/test-gpu0.json"
            write_json(old_status, {"state": "running", "worker": row})
            live_lease = FileLease(cp.parent / "locks/active.lock")
            live_lease.__enter__()
            calls, sleeps = [], []

            def tick(seconds):
                import time
                sleeps.append(seconds)
                if len(sleeps) == 1:
                    # Only the old worker can finish its live attempt. New code
                    # must wait, retaining pending job reservations meanwhile.
                    self.assertEqual(calls, [])
                    write_json(cp.parent / "jobs/active/result.json", {"state": "complete", "episodes": 1})
                    write_json(old_status, {"state": "waiting", "worker": row})
                    live_lease.__exit__()
                if len(sleeps) > 200:
                    raise AssertionError("Controller did not converge")
                original_sleep(.01)

            def fake_attempt(mp, pp, job, worker, inherited, output, development=False):
                calls.append((job["id"], development))
                if not development:
                    self.assertTrue((control / "qualification/complete.json").exists())
                    self.assertNotEqual(job["id"], "active")
                    self.assertTrue((failed / "result.before_memory_repair.json").exists())
                self.assertGreaterEqual(len(inherited), 2)
                folder = control / "qualification" / job["id"] if development else cp.parent / "jobs" / job["id"]
                folder.mkdir(parents=True, exist_ok=True)
                result = {"state": "complete", "episodes": len(job["episodes"])}
                write_json(folder / "result.json", result)
                return result

            def fake_report(path):
                (path.parent / "report").mkdir()
                write_json(path.parent / "report/complete.json", {"all_pass": True})

            import time
            original_sleep = time.sleep
            with patch.object(execution, "load", return_value=manifest), \
                 patch.object(execution, "load_plan", return_value=plan), \
                 patch.object(execution, "validate_worker", return_value=row), \
                 patch.object(execution, "attempt_job", side_effect=fake_attempt), \
                 patch.object(execution.time, "sleep", side_effect=tick), \
                 patch("medtoolbench.supplement.report.generate", side_effect=fake_report):
                execution.run(control / "manifest.json")
            self.assertEqual(calls, [("dev", True), ("oom", False)])
            self.assertTrue((control / "complete.json").exists())
            handoff = json.loads((control / "handoff/test-gpu0.json").read_text())
            self.assertFalse(handoff["model_process_interrupted"])


if __name__ == "__main__":
    unittest.main()
