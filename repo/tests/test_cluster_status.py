import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "scripts/cluster_status.py"
spec = importlib.util.spec_from_file_location("cluster_status_script", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ClusterStatusTests(unittest.TestCase):
    def test_saved_source_batches_not_lost_from_status_during_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, current = root / "attempt1", root / "attempt2"
            source.mkdir()
            current.mkdir()
            base = {"core_hashes": {"code": "same"}, "expected_keys": ["a", "b"]}
            (source / "config.json").write_text(json.dumps(base))
            (current / "config.json").write_text(json.dumps(dict(base, resume_from=str(source))))
            (source / "progress.json").write_text(json.dumps({"completed": 2}))
            (source / "audit.json").write_text(json.dumps({"errors": [], "unexpected": []}))
            self.assertEqual(module.retained_completed(current), 2)
            (current / "progress.json").write_text(json.dumps({"completed": 3}))
            self.assertEqual(module.retained_completed(current), 3)

    def test_invalid_source_not_credited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, current = root / "attempt1", root / "attempt2"
            source.mkdir()
            current.mkdir()
            (current / "config.json").write_text(json.dumps({"resume_from": str(source)}))
            (source / "progress.json").write_text(json.dumps({"completed": 100}))
            (source / "audit.json").write_text(json.dumps({"errors": ["bad"], "unexpected": []}))
            self.assertEqual(module.retained_completed(current), 0)


if __name__ == "__main__":
    unittest.main()
