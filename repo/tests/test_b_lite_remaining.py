import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import FileLease
from medtoolbench.research.job_state import write_json as atomic_json
from medtoolbench.supplement import b_lite_remaining as remaining
from medtoolbench.supplement import b_lite_transport as transport
from medtoolbench.supplement.b_lite import ROOT, SETTINGS


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def assignment():
    return {"jobs": [{"id": stage + "--qwen35_9b--000", "model": "qwen35_9b",
        "stage": stage, "episodes": [{"key": "fixed-cell"}], "settings": SETTINGS}
        for stage in ("qualification", "test")], "questions": [{"id": "unchanged"}]}


class RemainingQueueTests(unittest.TestCase):
    def test_exact_clone_no_reselection_or_previous_models(self):
        source = assignment()
        before = copy.deepcopy(source)
        jobs = remaining.clone_jobs(source)
        self.assertEqual(len(jobs), 4)
        self.assertEqual({j["model"] for j in jobs}, set(remaining.MODELS))
        self.assertTrue(all(j["settings"] == SETTINGS and j["episodes"] == [{"key": "fixed-cell"}] for j in jobs))
        jobs[0]["episodes"][0]["key"] = "modified"
        self.assertEqual(source, before)

    def test_sequential_models_and_inflight_drain(self):
        plan = {"jobs": remaining.clone_jobs(assignment())}
        first, second = remaining.MODELS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(remaining.selected_model(root, plan, set(), {}), first)
            self.assertEqual(remaining.selected_model(root, plan, {"preflight--" + first}, {first: "failed"}), first)
            self.assertEqual(remaining.selected_model(root, plan, {"test--" + first + "--000"}, {first: "failed"}), first)
            self.assertEqual(remaining.selected_model(root, plan, set(), {first: "failed"}), second)
            for job in plan["jobs"]:
                if job["model"] == first:
                    write_json(root / "jobs" / job["id"] / "result.json", {"state": "complete"})
            self.assertEqual(remaining.selected_model(root, plan, {"test--" + first + "--000"}, {}), first)
            self.assertEqual(remaining.selected_model(root, plan, set(), {}), second)
            for job in plan["jobs"]:
                write_json(root / "jobs" / job["id"] / "result.json", {"state": "complete"})
            self.assertIsNone(remaining.selected_model(root, plan, set(), {}))

    def test_selection_cannot_bypass_base_development_gate(self):
        plan = {"jobs": remaining.clone_jobs(assignment()), "plan_hash": "p"}
        with tempfile.TemporaryDirectory() as directory, patch.object(remaining, "BASE_ELIGIBLE", return_value=[]) as base:
            self.assertEqual(remaining.eligible(Path(directory), plan, set(), {}), [])
            base.assert_called_once()

    def test_drain_blocks_selection_and_attempt(self):
        plan = {"jobs": remaining.clone_jobs(assignment()), "plan_hash": "p"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(remaining.drain_requested(root, plan))
            write_json(root / "PAUSE_REQUESTED.json", {"action": "drain", "plan_hash": "p"})
            with patch.object(remaining, "BASE_ELIGIBLE") as choose, patch.object(remaining, "BASE_ATTEMPT") as launch, patch.object(remaining, "load", return_value=plan):
                self.assertEqual(remaining.eligible(root, plan, set(), {}), [])
                result = remaining.attempt(root, plan["jobs"][0], {}, None, None)
                self.assertEqual(result["state"], "not_dispatched")
                choose.assert_not_called()
                launch.assert_not_called()
            write_json(root / "PAUSE_REQUESTED.json", {"action": "drain", "plan_hash": "wrong"})
            with self.assertRaises(RuntimeError):
                remaining.drain_requested(root, plan)

    def test_late_gpu_waits_for_report_and_controller_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"workers": [{"gpu": i} for i in range(8)], "scheduling": {
                "late_gpu": 7, "dependency_control": str(root), "dependency_plan_hash": "llama"}}
            self.assertEqual([w["gpu"] for w in remaining.available_workers(plan)], list(range(7)))
            write_json(root / "complete.json", {"plan_hash": "llama", "all_pass": True, "formal_episodes": 1800})
            with self.assertRaises(RuntimeError):
                remaining.late_gpu_ready(plan)
            write_json(root / "report/metrics.json", {"complete": True, "plan_hash": "llama"})
            with FileLease(root / "supervisor.lock"):
                self.assertFalse(remaining.late_gpu_ready(plan))
            self.assertTrue(remaining.late_gpu_ready(plan))
            self.assertEqual(len(remaining.available_workers(plan)), 8)
            write_json(root / "complete.json", {"plan_hash": "other", "all_pass": True, "formal_episodes": 1800})
            with self.assertRaises(RuntimeError):
                remaining.late_gpu_ready(plan)

    def test_unreleased_gpu_cannot_launch(self):
        plan = {"plan_hash": "p", "scheduling": {"late_gpu": 7}}
        with tempfile.TemporaryDirectory() as directory, patch.object(remaining, "load", return_value=plan), patch.object(remaining, "late_gpu_ready", return_value=False), patch.object(remaining, "BASE_ATTEMPT") as launch:
            with self.assertRaises(RuntimeError):
                remaining.attempt(Path(directory), {}, {"gpu": 7}, None, None)
            launch.assert_not_called()


@unittest.skipUnless((ROOT / "data/models/gpt-oss-20b/config.json").is_file(), "local snapshot required")
class RealHarmonyTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.template = transport.tokenizer_for({"model_type": "gpt_oss", "path": str(ROOT / "data/models/gpt-oss-20b")})

    def test_real_tool_cycle_and_sorted_storage_replay(self):
        history = [{"role": "system", "content": "Original tools and task instructions"},
                   {"role": "user", "content": "Synthetic observation count only"}]
        frames = [
            '<|channel|>analysis<|message|>Inspect the interface.<|end|><|start|>assistant to=functions.metadata<|channel|>commentary<|message|>{}<|call|>',
            ' to=functions.get_resources<|channel|>commentary<|message|>{"resource_type":"Observation","patient_fhir_id":"synthetic","authoritative":true}<|call|>',
            r' to=functions.execute_python_code<|channel|>commentary<|message|>{"code":"print(len(retrieved_resources.get(\"Observation\", [])))"}<|call|>',
            '<|channel|>final<|message|>{"final":{"rows":[[1]]}}<|return|>']
        generations = []
        for raw in frames:
            ids = transport.render(self.tokenizer, history, "gpt_oss", self.template)
            text, status, meta = transport.decode(raw)
            self.assertNotEqual(status, "protocol_rejected", meta)
            generations.append({"raw_harmony": raw, "text": text, "harmony_status": status,
                "b_lite_native": meta, "input_tokens": len(ids), "input_prompt_hash": digest(ids)})
            history.append({"role": "assistant", "content": text, "b_lite_native": meta})
            if status == "native_tool":
                history.append({"role": "user", "content": '{"status":200,"count":1}'})
        row = json.loads(json.dumps({"messages": history, "generations": generations}, sort_keys=True))
        self.assertEqual(transport.audit_row(row, self.tokenizer, "gpt_oss", self.template),
                         ["native_tool"] * 3 + ["native_final"])
        rendered = self.tokenizer.decode(transport.render(self.tokenizer, history[:4], "gpt_oss", self.template), skip_special_tokens=False)
        self.assertIn("Inspect the interface.", rendered)
        self.assertIn("<|start|>functions.metadata to=assistant<|channel|>commentary", rendered)
        self.assertIn('"status": 200', rendered)
        self.assertTrue(rendered.endswith("<|start|>assistant"))


@unittest.skipUnless((ROOT / "data/models/Ministral-3-8B-Instruct-2512/config.json").is_file(), "local snapshot required")
class RealMinistralTemplateTests(unittest.TestCase):
    def test_exact_original_serialization(self):
        tok, _ = transport.tokenizer_for({"model_type": "mistral3", "path": str(ROOT / "data/models/Ministral-3-8B-Instruct-2512")})
        messages = [{"role": "system", "content": "Original public tools"},
                    {"role": "user", "content": "Synthetic test"},
                    {"role": "assistant", "content": '{"tool":"metadata"}'},
                    {"role": "user", "content": '{"status":200}'}]
        expected = tok.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        self.assertEqual(transport.render(tok, messages, "mistral3"), expected)


if __name__ == "__main__":
    unittest.main()
