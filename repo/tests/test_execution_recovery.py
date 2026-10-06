import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.research.build import save_gzip, load_gzip
from medtoolbench.research.fhir import HapiStore
from medtoolbench.research.input_validation import validate_path
from medtoolbench.research.job_state import dependency_state, wait_dependency, write_json
from medtoolbench.research.protocol import parse_action
from medtoolbench.research.runtime import GpuLease, batch_keys, recover_batches


class ExecutionRecoveryTests(unittest.TestCase):
    def test_invalid_model_url_returns_400_without_network(self):
        class Owner:
            port = 1234
        store = HapiStore(Owner())
        with patch("medtoolbench.research.fhir.urlopen") as request:
            for path in ("/Patient?birthdate=1943-05-19,name=Justin Gould", "/Patient?name=刘",
                         "/Patient?name=bad\nvalue", "/Patient?name=%0d%0a", "https://example.org/Patient", "/Patient/%xx"):
                self.assertEqual(store.call("GET", path)["status"], 400)
            request.assert_not_called()

    def test_encoded_values_and_root_pagination_are_accepted(self):
        for path in ("/Patient?name=Justin%20Gould", "/Patient?name=%E5%88%98", "/?_getpages=abc&_count=200"):
            self.assertTrue(validate_path(path))

    def test_invalid_fhir_body_does_not_reach_guard(self):
        for body in ([], {"subject": []}, {"subject": {"reference": []}}, {"value": float("nan")}):
            with self.assertRaises(ValueError):
                parse_action(json.dumps({"method": "POST", "path": "/Observation", "body": body}))

    def test_failed_campaign_dependency_is_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "progress.json", [{"model": "phi", "returncode": 1, "audit": None}])
            self.assertEqual(dependency_state(root, campaign=True)[0], "failed")
            with self.assertRaises(RuntimeError):
                wait_dependency(root, root, campaign=True, timeout=.1, poll=.01)
            self.assertEqual(json.loads((root / "status.json").read_text())["state"], "failed")

    def test_terminal_marker_overrides_stale_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "complete.json", {})
            write_json(root / "failed.json", {"reason": "corrupt audit"})
            self.assertEqual(dependency_state(root)[0], "failed")

    def test_gpu_lease_rejects_overlap_and_releases(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
            first, second = GpuLease(), GpuLease()
            first.path = second.path = Path(temp) / "gpu.lock"
            with first:
                with self.assertRaises(RuntimeError):
                    with second:
                        pass
            with second:
                pass

    def test_only_complete_batches_are_recovered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "old", root / "new"
            for d in (source / "episodes", source / "states", output / "episodes"):
                d.mkdir(parents=True)
            config = {"model": "same", "core_hashes": {"run": "same"}}
            write_json(source / "config.json", config)
            batches = [[({"id": x}, "clean", "vanilla") for x in pair] for pair in (("a", "b"), ("c", "d"))]
            save_gzip(source / "states/initial.json.gz", [])
            for key in batch_keys(batches[0]) + batch_keys(batches[1])[:1]:
                save_gzip(source / "episodes" / (key + ".json.gz"), {"before_ref": "states/initial.json.gz",
                    "grade": {"success": True, "unsafe": []}, "triggered": 0, "stop": "final"})
            with patch("medtoolbench.research.audit.audit", return_value={"errors": [], "unexpected": []}):
                summaries, remaining, proof = recover_batches(source, output, root, config, batches)
            self.assertEqual(len(summaries), 2)
            self.assertEqual(remaining, [batches[1]])
            self.assertEqual(len(proof["recovered_episode_keys"]), 2)
            self.assertFalse((output / "episodes" / (batch_keys(batches[1])[0] + ".json.gz")).exists())

    def test_code_change_prevents_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "config.json", {"core_hashes": {"run": "old"}})
            with self.assertRaisesRegex(RuntimeError, "core_hashes"):
                recover_batches(root, root, root, {"core_hashes": {"run": "new"}}, [])

    def test_gzip_commit_is_atomic_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a.gz", Path(temp) / "b.gz"
            save_gzip(a, {"value": 1})
            save_gzip(b, {"value": 1})
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(load_gzip(a), {"value": 1})
            self.assertFalse(list(Path(temp).glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
