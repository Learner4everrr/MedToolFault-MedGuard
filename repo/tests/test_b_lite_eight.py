import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.supplement import b_lite_eight as eight


class EightGpuResumeTests(unittest.TestCase):
    def scientific_fixture(self):
        plan = {key: {"fixed": key} for key in eight.SCIENCE_KEYS}
        plan["systems"] = [{"name": "llama31_8b", "batch_size": 2}]
        plan["source_hashes"] = {"frozen.py": "unchanged"}
        return plan

    def test_resource_change_does_not_change_science(self):
        old = self.scientific_fixture()
        new = copy.deepcopy(old)
        new["workers"] = list(range(8))
        new["source_hashes"]["resource_only.py"] = "new"
        eight.assert_science(new, old)

    def test_budgets_questions_adapter_and_models_cannot_change(self):
        old = self.scientific_fixture()
        for key in eight.SCIENCE_KEYS:
            new = copy.deepcopy(old)
            new[key] = "changed"
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                eight.assert_science(new, old)

    def test_frozen_source_cannot_change(self):
        old = self.scientific_fixture()
        new = copy.deepcopy(old)
        new["source_hashes"]["frozen.py"] = "modified"
        with self.assertRaises(RuntimeError):
            eight.assert_science(new, old)

    def interruption(self):
        return {"state": "quarantined", "execution": {"timeout": "controller_stop"},
                "plan_hash": "old", "job": "j", "partial_rows_counted": False}

    def test_only_administrative_interruption_can_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            result, job = self.interruption(), {"id": "j", "stage": "test"}
            eight.administrative_interruption(result, Path(temp), job)
            for reason in (None, "wall_timeout", "no_progress_timeout", "cuda_oom"):
                result["execution"]["timeout"] = reason
                with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                    eight.administrative_interruption(result, Path(temp), job)

    def test_audited_or_completed_shard_cannot_be_restart_candidate(self):
        for name in ("complete.json", "audit.json", "transport_audit.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                (output / name).touch()
                with self.assertRaises(RuntimeError):
                    eight.administrative_interruption(self.interruption(), output, {"id": "j", "stage": "test"})

    def test_development_is_never_restarted(self):
        with self.assertRaises(RuntimeError):
            eight.administrative_interruption(self.interruption(), Path("unused"), {"id": "j", "stage": "qualification"})

    def test_imported_result_uses_original_plan_and_exact_path(self):
        job = {"id": "j", "stage": "qualification"}
        prior = {"jobs": [job], "plan_hash": "old"}
        plan = {"resume": {"adopted_shards": {"j": {"run": "/old/run", "files": {},
                                                   "previous_plan_hash": "old"}}}}
        with patch.object(eight, "read", return_value=prior), patch.object(eight, "OLD_VERIFY") as verify:
            eight.verify_completion(Path("/old/run"), plan, job)
            verify.assert_called_once_with(Path("/old/run"), prior, job)
            with self.assertRaises(RuntimeError):
                eight.verify_completion(Path("/old/different_run"), plan, job)

    def test_old_partial_path_cannot_be_imported(self):
        plan = {"resume": {"adopted_shards": {}, "control": "/new"}}
        with patch.object(eight, "OLD_VERIFY") as verify:
            with self.assertRaises(RuntimeError):
                eight.verify_completion(Path("/old/partial"), plan, {"id": "j"})
            verify.assert_not_called()

    def test_durable_drain_blocks_eligibility(self):
        plan = {"plan_hash": "frozen"}
        with patch.object(eight, "read", return_value={"action": "drain", "plan_hash": "frozen"}), \
                patch.object(eight, "OLD_ELIGIBLE") as eligible:
            self.assertEqual(eight.eligible(Path("unused"), plan, set(), {}), [])
            eligible.assert_not_called()

    def test_drain_rechecked_at_attempt_before_subprocess(self):
        job = {"id": "j", "model": "llama31_8b", "stage": "test"}
        with patch.object(eight, "load", return_value={"plan_hash": "frozen"}), \
                patch.object(eight, "drain_requested", return_value=True), patch.object(eight, "OLD_ATTEMPT") as attempt:
            self.assertEqual(eight.attempt(Path("unused"), job, {}, 1, None)["state"], "not_dispatched")
            attempt.assert_not_called()

    def test_corrupt_drain_fails_closed(self):
        with patch.object(eight, "read", return_value={"action": "drain", "plan_hash": "other"}):
            with self.assertRaises(RuntimeError):
                eight.drain_requested(Path("unused"), {"plan_hash": "frozen"})

    def test_adopted_job_never_reexecutes(self):
        job = {"id": "j", "model": "llama31_8b", "stage": "test"}
        plan = {"resume": {"adopted_shards": {"j": {}}}}
        with patch.object(eight, "load", return_value=plan), patch.object(eight, "drain_requested", return_value=False), \
                patch.object(eight, "OLD_ATTEMPT") as attempt:
            with self.assertRaises(RuntimeError):
                eight.attempt(Path("unused"), job, {}, 1, None)
            attempt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
