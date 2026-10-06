import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from medtoolbench.research.job_state import write_json
from medtoolbench.research.protocol import parse_action as original_parse
from medtoolbench.research.run import Episode
from medtoolbench.supplement.input_guard import parse_action, validate_meta
from medtoolbench.supplement.isolated_execution import (
    IntegrityError, eligible_jobs, failure_kind, model_gate, pending_ledger,
    validate_retry_history, attempt_job)
from medtoolbench.supplement.serial_model import SerialGemma


class InputGuardTests(unittest.TestCase):
    def test_null_and_other_non_objects_rejected_without_mutation(self):
        for meta in (None, [], "x", 0, False):
            action = {"method": "POST", "path": "/ServiceRequest", "body": {
                "resourceType": "ServiceRequest", "subject": {"reference": "Patient/p"}, "meta": meta}}
            old = copy.deepcopy(action)
            with self.assertRaisesRegex(ValueError, "body.meta"):
                validate_meta(action)
            self.assertEqual(action, old)
            with self.assertRaises(ValueError):
                parse_action(json.dumps(action))

    def test_valid_payloads_exactly_unchanged(self):
        bodies = [{"resourceType": "Patient"}, {"resourceType": "Patient", "meta": {}},
                  {"resourceType": "Patient", "meta": {"profile": ["urn:profile"], "tag": [{"code": "x"}]}}]
        for body in bodies:
            raw = json.dumps({"method": "POST", "path": "/Patient", "body": body})
            self.assertEqual(parse_action(raw), original_parse(raw))

    def test_final_answers_not_misclassified_as_fhir_bodies(self):
        raw = '{"final":{"body":{"meta":null}}}'
        self.assertEqual(parse_action(raw), original_parse(raw))

    def test_bad_meta_never_dispatches_under_any_policy_and_feedback_is_logged(self):
        from medtoolbench.supplement.transport import CONDITIONS, CallBudget, FactorClient
        raw = '{"method":"POST","path":"/ServiceRequest","body":{"resourceType":"ServiceRequest","subject":{"reference":"Patient/p"},"meta":null}}'
        for condition in CONDITIONS:
            dispatch = Mock()
            budget = CallBudget(dispatch, 48)
            episode = Episode.__new__(Episode)
            episode.client = FactorClient(budget, condition)
            episode.messages, episode.generations, episode.parsed_actions = [], [], []
            with patch("medtoolbench.research.run.parse_action", parse_action):
                episode.advance({"text": raw})
            dispatch.assert_not_called()
            self.assertEqual(budget.used, 0)
            self.assertEqual(episode.messages[0]["content"], raw)
            self.assertEqual(json.loads(episode.messages[-1]["content"])["status"], 400)
            self.assertEqual(len(episode.generations), 1)

    def test_unrelated_attribute_error_not_swallowed(self):
        episode = Episode.__new__(Episode)
        episode.client = Mock()
        episode.client.call.side_effect = AttributeError("unrelated code bug")
        episode.messages, episode.generations, episode.parsed_actions = [], [], []
        with patch("medtoolbench.research.run.parse_action", parse_action), self.assertRaises(AttributeError):
            episode.advance({"text": '{"method":"GET","path":"/metadata"}'})


class SerialExecutionTests(unittest.TestCase):
    def test_greedy_preserves_order_and_budget_while_splitting_physical_batches(self):
        model = SerialGemma.__new__(SerialGemma)
        seen = []
        def generate(instance, prompts, limit):
            seen.append((prompts, limit))
            return [{"token": prompts[0][0], "batch_size": len(prompts)}]
        with patch("medtoolbench.supplement.runner.SampleModel.generate", generate):
            result = model.generate([[11], [22], [33]], 4096)
        self.assertEqual([r["token"] for r in result], [11, 22, 33])
        self.assertEqual(seen, [([[11]], 4096), ([[22]], 4096), ([[33]], 4096)])

    def test_sampling_preserves_temperature_and_does_not_reseed(self):
        model = SerialGemma.__new__(SerialGemma)
        seen = []
        def generate(instance, prompts, limit, temperature):
            seen.append((prompts, limit, temperature))
            return [prompts[0]]
        with patch("medtoolbench.supplement.runner.SampleModel.generate_sample", generate):
            result = model.generate_sample([[1], [2]], 2048, .7)
        self.assertEqual(result, [[1], [2]])
        self.assertEqual(seen, [([[1]], 2048, .7), ([[2]], 2048, .7)])


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.control = self.root / "control"
        self.control.mkdir()
        self.cp = self.root / "a/plan.json"
        self.jobs = [self.job("dev-good", "healthy", "qualification"),
                     self.job("test-good", "healthy"), self.job("dev-bad", "gemma3_12b", "qualification"),
                     self.job("test-bad", "gemma3_12b")]
        self.plan = {"jobs": self.jobs, "plan_hash": "p", "require_oracle": False, "python": "unused"}
        self.manifest = {"manifest_hash": "m"}

    def tearDown(self):
        self.tmp.cleanup()

    def job(self, name, model, stage="test"):
        return {"id": name, "model": model, "stage": stage, "episodes": [{"key": name}]}

    def completed(self, job):
        folder = self.cp.parent / "jobs" / job["id"]
        out = folder / "attempt1"
        out.mkdir(parents=True)
        write_json(out / "config.json", {"plan_hash": "p", "job": job})
        write_json(out / "complete.json", {"all_pass": True})
        write_json(out / "audit.json", {"all_pass": True, "episodes": 1, "plan_hash": "p"})
        write_json(folder / "result.json", {"state": "complete", "episodes": 1, "run": str(out)})
        return out

    def test_other_model_qualification_failure_does_not_block_healthy_model(self):
        self.completed(self.jobs[0])
        folder = self.cp.parent / "jobs" / "dev-bad"
        folder.mkdir(parents=True)
        write_json(folder / "result.json", {"state": "quarantined"})
        self.assertEqual(model_gate(self.cp, self.plan, "healthy"), "passed")
        self.assertEqual(model_gate(self.cp, self.plan, "gemma3_12b"), "failed")
        candidates = eligible_jobs([(self.cp, self.plan)], self.control, self.manifest,
                                   {"gemma3_12b": {}}, set(), False)
        self.assertEqual([r[3]["id"] for r in candidates], ["test-good"])

    def test_own_qualification_still_blocks_test(self):
        candidates = eligible_jobs([(self.cp, self.plan)], self.control, self.manifest, {}, set(), False)
        self.assertEqual([r[3]["id"] for r in candidates], ["dev-good"])

    def test_shared_oracle_failure_blocks_held_out(self):
        self.completed(self.jobs[0])
        self.plan["require_oracle"] = True
        self.assertEqual(model_gate(self.cp, self.plan, "healthy"), "waiting")
        oracle = self.cp.parent / "oracle"
        oracle.mkdir()
        write_json(oracle / "failed.json", {"all_pass": False})
        self.assertEqual(model_gate(self.cp, self.plan, "healthy"), "failed")

    def test_complete_and_active_jobs_never_selected(self):
        self.completed(self.jobs[0])
        key = (str(self.cp), "test-good", False)
        self.assertEqual(eligible_jobs([(self.cp, self.plan)], self.control, self.manifest, {}, {key}, False), [])

    def test_failures_remain_in_denominator(self):
        self.completed(self.jobs[0])
        pending = pending_ledger([(self.cp, self.plan)], self.control)
        self.assertEqual(len(pending), 3)
        result = json.loads((self.control / "pending.json").read_text())
        self.assertEqual(result["audited_episodes_including_development"] + result["pending_episodes_including_development"], 4)
        self.assertFalse(result["failed_partial_rows_counted"])

    def failed_history(self, error="torch.OutOfMemoryError: CUDA out of memory"):
        folder = self.root / "failed"
        (folder / "attempt1").mkdir(parents=True)
        write_json(folder / "attempt1/failure.json", {"error": error})
        write_json(folder / "result.json", {"state": "failed"})
        return folder

    def test_known_mixed_oom_and_meta_failure_history_is_repairable(self):
        folder = self.failed_history()
        (folder / "attempt2").mkdir()
        write_json(folder / "attempt2/failure.json", {"error": "state_payload AttributeError: 'NoneType' object has no attribute 'pop'"})
        self.assertEqual(len(validate_retry_history(folder)["prior_attempts"]), 2)

    def test_unknown_failure_requires_review(self):
        with self.assertRaises(IntegrityError):
            validate_retry_history(self.failed_history("grader mismatch"))

    def test_unresolved_or_completed_attempt_never_retried(self):
        folder = self.failed_history()
        write_json(folder / "attempt1/audit.json", {"all_pass": False})
        with self.assertRaises(IntegrityError):
            validate_retry_history(folder)

    def test_failure_classification_does_not_consult_accuracy(self):
        self.assertEqual(failure_kind("success_rate=0"), "unclassified_infrastructure")
        self.assertEqual(failure_kind("torch.OutOfMemoryError: CUDA out of memory"), "cuda_oom")

    def test_child_failure_is_quarantined_not_raised_and_no_blind_retry(self):
        job = self.jobs[1]
        worker = {"name": "gpu0", "gpu": 0}
        (self.control / "workers").mkdir()
        def child(cmd, *args, **kwargs):
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True)
            write_json(out / "failure.json", {"error": "torch.OutOfMemoryError: CUDA out of memory"})
            return {"returncode": 1, "timeout": None}
        with patch("medtoolbench.supplement.isolated_execution.load", return_value=self.manifest), \
             patch("medtoolbench.supplement.isolated_execution.load_plan", return_value=self.plan), \
             patch("medtoolbench.supplement.isolated_execution.execute", side_effect=child) as execute:
            result = attempt_job(self.control / "manifest.json", self.cp, job, worker, [])
            self.assertEqual(result["state"], "quarantined")
            self.assertEqual(result["assigned_episodes"], 1)
            self.assertEqual(execute.call_count, 1)
            with self.assertRaises(IntegrityError):
                attempt_job(self.control / "manifest.json", self.cp, job, worker, [])
            self.assertEqual(execute.call_count, 1)

    def test_completed_successful_job_is_never_retried_for_low_accuracy(self):
        self.completed(self.jobs[1])
        with self.assertRaises(IntegrityError):
            validate_retry_history(self.cp.parent / "jobs" / self.jobs[1]["id"])

    def test_gemma_cannot_bypass_shadow_gate(self):
        self.completed(self.jobs[2])
        with patch("medtoolbench.supplement.isolated_execution.preflight_passes", return_value=True):
            self.assertEqual(eligible_jobs([(self.cp, self.plan)], self.control, self.manifest, {}, set(), True), [])

    def test_qualification_audit_identity_is_checked(self):
        out = self.completed(self.jobs[0])
        write_json(out / "audit.json", {"all_pass": True, "episodes": 1, "plan_hash": "other"})
        with self.assertRaises(RuntimeError):
            model_gate(self.cp, self.plan, "healthy")

    def test_controller_continues_healthy_test_after_gemma_shadow_oom(self):
        import time
        from medtoolbench.supplement.isolated_execution import run
        self.completed(self.jobs[2])
        rows = [{"name": "gpu" + str(i), "gpu": i} for i in range(8)]
        self.plan["workers"] = rows
        self.manifest.update(campaigns=[str(self.cp)], legacy_worker_lock_roots=[])
        write_json(self.control / "handoff.json", {"all_legacy_controllers_stopped": True, "manifest_hash": "m"})
        calls = []
        def child(cmd, *args, **kwargs):
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True)
            if "--mode" in cmd:
                write_json(out / "complete.json", {"all_pass": True, "manifest_hash": "m"})
                return {"returncode": 0}
            job = next(j for j in self.jobs if j["id"] == cmd[cmd.index("--job") + 1])
            calls.append(job["id"])
            if job["model"] == "gemma3_12b":
                write_json(out / "failure.json", {"error": "torch.OutOfMemoryError: CUDA out of memory"})
                return {"returncode": 1}
            write_json(out / "config.json", {"plan_hash": "p", "job": job})
            write_json(out / "audit.json", {"all_pass": True, "episodes": 1, "plan_hash": "p"})
            write_json(out / "complete.json", {"all_pass": True})
            return {"returncode": 0}
        real_sleep = time.sleep
        with patch("medtoolbench.supplement.isolated_execution.load", return_value=self.manifest), \
             patch("medtoolbench.supplement.isolated_execution.load_plan", return_value=self.plan), \
             patch("medtoolbench.supplement.isolated_execution.validate_worker"), \
             patch("medtoolbench.supplement.isolated_execution.require_cpu_validation", return_value={"behavior_unchanged_for_saved_trajectories": True}), \
             patch("medtoolbench.supplement.isolated_execution.execute", side_effect=child), \
             patch("medtoolbench.supplement.isolated_execution.time.sleep", side_effect=lambda n: real_sleep(.01)):
            run(self.control / "manifest.json")
        self.assertEqual(calls.count("dev-bad"), 1)
        self.assertIn("test-good", calls)
        self.assertNotIn("test-bad", calls)
        state = json.loads((self.control / "status.json").read_text())
        self.assertEqual(state["state"], "needs_review")
        self.assertIn("gemma3_12b", state["blocked_models"])
        self.assertFalse((self.control / "integrity_failure.json").exists())
        self.assertFalse((self.control / "complete.json").exists())


if __name__ == "__main__":
    unittest.main()
