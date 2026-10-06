import copy
import unittest

from medtoolbench.supplement import b_lite_gptoss_env as recovery


def fixture():
    old = {"plan_hash": "previous", "python": "original-runtime", "questions": ["same-question"],
        "systems": [{"name": "ministral3_8b"}, {"name": "gptoss20b", "batch_size": 2, "revision": "fixed"}],
        "jobs": [{"id": "q-min", "model": "ministral3_8b", "stage": "qualification"},
                 {"id": "q-gpt", "model": "gptoss20b", "stage": "qualification", "settings": {"max_turns": 20}},
                 {"id": "t-gpt", "model": "gptoss20b", "stage": "test", "episodes": ["fixed-cell"]}],
        "workers": [{"gpu": i} for i in range(8)], "statistics": {"bootstrap": 10000},
        "source_hashes": {"original.py": "codehash"}, "input_hashes": {"data.json": "datahash"},
        "scheduling": {"model_order": ["ministral3_8b", "gptoss20b"], "initial_gpus": list(range(7)),
                       "late_gpu": 7, "dependency_plan_hash": "llama"}}
    snapshot = {"prefix": str(recovery.ROOT / "env_gptoss"), "versions": dict(recovery.REQUIRED),
                "kernels_available": True, "triton_available": True}
    return old, snapshot


class GptOssEnvironmentRecoveryTests(unittest.TestCase):
    def test_exact_gpt_only_matrix_and_original_untouched(self):
        old, snapshot = fixture()
        before = copy.deepcopy(old)
        plan = recovery.recovered_plan(old, snapshot)
        recovery.assert_science(plan, old)
        self.assertEqual(old, before)
        self.assertEqual([j["id"] for j in plan["jobs"]], ["q-gpt", "t-gpt"])
        self.assertEqual(plan["python"], str(recovery.PYTHON))
        self.assertFalse(plan["runtime_recovery"]["prior_models_rerun"])

    def test_changes_to_science_are_rejected(self):
        old, snapshot = fixture()
        for key in ("questions", "statistics", "workers"):
            with self.subTest(key=key):
                plan = recovery.recovered_plan(old, snapshot)
                plan[key] = []
                with self.assertRaises(RuntimeError):
                    recovery.assert_science(plan, old)

    def test_cannot_rerun_other_models_or_change_gpt_budget(self):
        old, snapshot = fixture()
        plan = recovery.recovered_plan(old, snapshot)
        plan["jobs"][0]["settings"]["max_turns"] = 10
        with self.assertRaises(RuntimeError):
            recovery.assert_science(plan, old)
        plan = recovery.recovered_plan(old, snapshot)
        plan["jobs"].append(old["jobs"][0])
        with self.assertRaises(RuntimeError):
            recovery.assert_science(plan, old)

    def test_frozen_code_and_data_cannot_be_rewritten(self):
        old, snapshot = fixture()
        for field in ("source_hashes", "input_hashes"):
            plan = recovery.recovered_plan(old, snapshot)
            plan[field] = {}
            with self.assertRaises(RuntimeError):
                recovery.assert_science(plan, old)

    def test_runtime_wrong_environment_or_dependencies_rejected(self):
        _, snapshot = fixture()
        recovery.check_runtime(snapshot)
        for field, value in (("prefix", "/other/environment"), ("kernels_available", False), ("triton_available", False)):
            bad = copy.deepcopy(snapshot)
            bad[field] = value
            with self.assertRaises(RuntimeError):
                recovery.check_runtime(bad)
        bad = copy.deepcopy(snapshot)
        bad["versions"]["kernels"] = "0.11.5"
        with self.assertRaises(RuntimeError):
            recovery.check_runtime(bad)

    def test_only_zero_episode_dependency_failure_qualifies(self):
        result = {"state": "quarantined", "stage": "preflight", "model": recovery.MODEL,
            "plan_hash": "old", "assigned_episodes": 0, "partial_rows_counted": False,
            "failure": {"error": "MXFP4 dependencies invalid; BF16 fallback is forbidden"}}
        recovery.check_failure(result, "old")
        for key, value in (("assigned_episodes", 1), ("stage", "test"), ("plan_hash", "other"),
                           ("failure", {"error": "CUDA out of memory"})):
            bad = copy.deepcopy(result)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                recovery.check_failure(bad, "old")

    def test_full_suite_and_actual_gpt_runtime_are_both_required(self):
        commands = dict(recovery.validation_commands({"python": str(recovery.PYTHON)}))
        self.assertEqual(commands["full_cpu"][0], str(recovery.ROOT / "env_modern/bin/python"))
        self.assertIn("discover", commands["full_cpu"])
        target = commands["target_runtime"]
        self.assertEqual(target[0], str(recovery.PYTHON))
        self.assertIn("test_b_lite_remaining.RealHarmonyTemplateTests", target)
        self.assertIn("test_b_lite_gptoss_env", target)
        self.assertIn("test_gptoss_compile", target)


if __name__ == "__main__":
    unittest.main()
