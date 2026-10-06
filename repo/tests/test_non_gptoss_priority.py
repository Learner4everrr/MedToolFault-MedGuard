import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.supplement import non_gptoss_priority as priority
from medtoolbench.supplement.storage_recovery import atomic_json, read


class OtherModelPriorityTests(unittest.TestCase):
    def test_only_declared_other_models_allowed(self):
        for model in priority.ALLOWED:
            priority.require_allowed(model)
        for model in ('gptoss20b', 'gemma3_12b', 'unknown'):
            with self.assertRaises(RuntimeError):
                priority.require_allowed(model)

    def test_deferred_models_never_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [{'id': model, 'model': model, 'stage': 'qualification'}
                    for model in (*priority.ALLOWED, *priority.DEFERRED, 'unknown')]
            rows = priority.eligible([(Path(tmp) / 'plan.json', {'jobs': jobs})], set(), {})
            self.assertEqual({r[4]['model'] for r in rows}, priority.ALLOWED)

    def test_direct_dispatch_rejects_deferred_before_loading_or_launching(self):
        with patch.object(priority, '_ATTEMPT') as launch:
            for model in priority.DEFERRED:
                with self.assertRaises(RuntimeError):
                    priority.attempt(None, None, {'model': model}, None, None, None)
            launch.assert_not_called()

    def test_completed_or_started_jobs_never_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'plan.json'
            jobs = [{'id': name, 'model': 'qwen35_9b', 'stage': 'qualification'}
                    for name in ('complete', 'started', 'fresh')]
            atomic_json(cp.parent / 'jobs/complete/result.json', {'state': 'complete'})
            (cp.parent / 'jobs/started/attempt1').mkdir(parents=True)
            rows = priority.eligible([(cp, {'jobs': jobs})], set(), {})
            self.assertEqual([r[4]['id'] for r in rows], ['fresh'])

    def test_existing_development_gate_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                'medtoolbench.supplement.gptoss_pipeline.model_gate', return_value='waiting'):
            cp = Path(tmp) / 'plan.json'
            job = {'id': 'test', 'model': 'qwen35_9b', 'stage': 'test'}
            self.assertEqual(priority.eligible([(cp, {'jobs': [job]})], set(), {}), [])

    def test_quarantined_model_does_not_block_healthy_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [{'id': model, 'model': model, 'stage': 'qualification'}
                    for model in ('qwen35_9b', 'ministral3_8b')]
            rows = priority.eligible([(Path(tmp) / 'plan.json', {'jobs': jobs})], set(), {'qwen35_9b': {}})
            self.assertEqual([r[4]['model'] for r in rows], ['ministral3_8b'])

    def test_completion_is_scoped_and_never_resumes_deferred(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(priority, 'verify_completion', return_value={'episodes': 4}):
            cp = Path(tmp) / 'original/plan.json'
            job = {'id': 'done', 'model': 'qwen35_9b', 'stage': 'test'}
            atomic_json(cp.parent / 'jobs/done/result.json', {'state': 'complete', 'run': 'read-only-run'})
            path = Path(tmp) / 'control/manifest.json'
            priority.finalize(path, {'manifest_hash': 'hash'}, [(cp, {'jobs': [job]})])
            proof = read(path.parent / 'complete.json')
            self.assertFalse(proof['entire_project_complete'])
            self.assertFalse(proof['automatic_deferred_resume'])
            self.assertEqual(proof['formal_episodes'], 4)

    def test_incomplete_cohort_cannot_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'original/plan.json'
            job = {'id': 'missing', 'model': 'qwen35_9b', 'stage': 'test'}
            with self.assertRaises(RuntimeError):
                priority.finalize(Path(tmp) / 'manifest.json', {}, [(cp, {'jobs': [job]})])


if __name__ == '__main__':
    unittest.main()
