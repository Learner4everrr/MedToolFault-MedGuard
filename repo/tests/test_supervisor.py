import json
import tempfile
import unittest
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.freeze_study import matrix, CONDITIONS
from medtoolbench.research.job_state import write_json
from medtoolbench.research.supervisor import command, load_plan, ready_completion


class SupervisorTests(unittest.TestCase):
    def test_shards_cover_each_test_task_exactly_once_per_model(self):
        tasks = [{"id": str(i), "split": "test", "hazards": ["clean", "transient_503"]} for i in range(25)]
        tasks += [{"id": "dev-only", "split": "dev", "hazards": ["clean"]}]
        jobs = matrix(tasks, [{"name": "a"}, {"name": "b"}], shards=8)
        for model in ("a", "b"):
            ids = [i for j in jobs if j["model"] == model for i in j["task_ids"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {str(i) for i in range(25)})
        self.assertEqual(sum(j["expected_episodes"] for j in jobs), 25 * 2 * 5 * 2)

    def test_modified_frozen_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {"a": 1}
            plan["plan_hash"] = digest(plan)
            write_json(root / "plan.json", plan)
            self.assertEqual(load_plan(root), plan)
            plan["a"] = 2
            write_json(root / "plan.json", plan)
            with self.assertRaises(RuntimeError):
                load_plan(root)

    def test_incomplete_or_failed_job_cannot_complete_study(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {"jobs": [{"id": "a"}], "expected_episodes": 5, "plan_hash": "frozen"}
            self.assertIsNone(ready_completion(root, plan))
            folder = root / "jobs/a"
            folder.mkdir(parents=True)
            write_json(folder / "result.json", {"state": "failed", "episodes": 0, "run": "bad"})
            self.assertIsNone(ready_completion(root, plan))
            write_json(folder / "result.json", {"state": "complete", "episodes": 5, "run": "good"})
            self.assertEqual(ready_completion(root, plan)["episodes"], 5)

    def test_command_locks_shard_budget_conditions_and_reasoning(self):
        plan = {"benchmark": "/b", "validation": "/v", "image": "/i", "conditions": CONDITIONS,
                "generation": {"max_turns": 12, "max_new_tokens": 2048, "max_context": 16384, "seed": 42}}
        job = {"task_shards": 8, "task_shard": 3}
        system = {"path": "/m", "thinking": True, "batch_size": 2}
        cmd = command(plan, job, system, Path("/new"), Path("/old"))
        self.assertEqual(cmd[cmd.index("--split") + 1], "test")
        self.assertEqual(cmd[cmd.index("--max-new-tokens") + 1], "2048")
        self.assertEqual(cmd[cmd.index("--task-shard") + 1], "3")
        self.assertIn("--thinking", cmd)
        self.assertEqual(cmd[-2:], ["--resume-from", "/old"])


if __name__ == "__main__":
    unittest.main()
