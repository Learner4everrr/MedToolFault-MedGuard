import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.supplement import concurrent_pipeline as pipeline
from medtoolbench.supplement.storage_recovery import atomic_json, read


class ConcurrentPipelineTests(unittest.TestCase):
    def job(self, name, concurrent=False, model='qwen35_9b'):
        return {'id': name, 'model': model, 'stage': 'qualification', 'episodes': [{'key': name}],
                'settings': {'stress': {'concurrent': concurrent}}}

    def test_failed_original_remains_but_only_repair_overlay_is_scheduled(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp, control = Path(tmp) / 'original/plan.json', Path(tmp) / 'control'
            job = self.job('repair', True)
            jobs = [job, self.job('healthy'), self.job('gpt', model='gptoss20b')]
            m = {'repair_jobs': {pipeline.repair_key(cp, job['id']): {}}}
            source = cp.parent / 'jobs/repair/result.json'
            atomic_json(source, {'state': 'quarantined'})
            pending, blocked, completed = pipeline.inventory(control, [(cp, {'jobs': jobs})], m)
            self.assertEqual((len(pending), blocked, completed), (2, {}, 0))
            ready = pipeline.eligible(control, [(cp, {'jobs': jobs})], m, set(), {})
            self.assertEqual({r[5]['id'] for r in ready}, {'repair', 'healthy'})
            self.assertEqual(read(source)['state'], 'quarantined')

    def test_failed_repair_blocks_and_never_blindly_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp, control = Path(tmp) / 'original/plan.json', Path(tmp) / 'control'
            job = self.job('repair', True)
            m = {'repair_jobs': {pipeline.repair_key(cp, job['id']): {}}}
            folder = pipeline.folder_for(control, cp, job, m)
            atomic_json(folder / 'result.json', {'state': 'quarantined'})
            plans = [(cp, {'jobs': [job, self.job('healthy')]})]
            _, blocked, _ = pipeline.inventory(control, plans, m)
            self.assertEqual(pipeline.eligible(control, plans, m, set(), blocked), [])
            self.assertIn('qwen35_9b', blocked)

    def test_saved_partial_repair_cannot_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp, control = Path(tmp) / 'plan.json', Path(tmp) / 'control'
            job = self.job('repair', True)
            m = {'repair_jobs': {pipeline.repair_key(cp, 'repair'): {}}}
            (pipeline.folder_for(control, cp, job, m) / 'attempt1').mkdir(parents=True)
            self.assertEqual(pipeline.eligible(control, [(cp, {'jobs': [job]})], m, set(), {}), [])

    def test_completed_old_concurrent_never_reselected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp, control = Path(tmp) / 'plan.json', Path(tmp) / 'control'
            job = self.job('old', True)
            atomic_json(cp.parent / 'jobs/old/result.json', {'state': 'complete'})
            m = {'repair_jobs': {}}
            self.assertEqual(pipeline.eligible(control, [(cp, {'jobs': [job]})], m, set(), {}), [])
            self.assertEqual(pipeline.inventory(control, [(cp, {'jobs': [job]})], m), ([], {}, 1))

    def test_development_gate_and_live_ownership_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp, control = Path(tmp) / 'plan.json', Path(tmp) / 'control'
            job = self.job('repair', True)
            m = {'repair_jobs': {pipeline.repair_key(cp, 'repair'): {}}}
            plans = [(cp, {'jobs': [job]})]
            self.assertEqual(pipeline.eligible(control, plans, m, {(str(cp), 'repair')}, {}), [])
            job['stage'] = 'test'
            with patch.object(pipeline, 'model_gate', return_value='waiting'):
                self.assertEqual(pipeline.eligible(control, plans, m, set(), {}), [])

    def test_missing_jobs_prevent_complete_even_with_prior_failed_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp, control = Path(tmp) / 'plan.json', Path(tmp) / 'control'
            job = self.job('repair', True)
            m = {'repair_jobs': {pipeline.repair_key(cp, 'repair'): {}}}
            with self.assertRaisesRegex(RuntimeError, 'Unresolved'):
                pipeline.finalize(control, [(cp, {'jobs': [job]})], m)

    def proof(self):
        policies = ('vanilla', 'informed', 'evidence_only', 'medguard', 'no_read', 'no_write', 'no_retry')
        hazards = ('commit_timeout', 'no_commit_timeout')
        return {'all_pass': True, 'held_out_tasks_used': False, 'model_inference': False,
            'clinical_accuracy_not_a_gate': True, 'source_hashes': {},
            'old_crashes_reproduced': ['encoded_date', 'resource_mismatch'],
            'checks': [{'policy': p, 'hazard': h, 'malformed': b, 'all_pass': True}
                for p in policies for h in hazards for b in ('encoded_date', 'resource_mismatch')],
            'positive_path_equivalence': [{'policy': p, 'hazard': h, 'all_pass': True}
                for p in policies for h in hazards]}

    def test_transport_gate_requires_exact_complete_coverage(self):
        good = self.proof()
        pipeline.check_validation(good)
        for field in ('checks', 'positive_path_equivalence', 'old_crashes_reproduced'):
            broken = copy.deepcopy(good)
            broken[field].pop()
            with self.assertRaises(RuntimeError):
                pipeline.check_validation(broken)
        duplicate = copy.deepcopy(good)
        duplicate['checks'][0] = duplicate['checks'][1]
        with self.assertRaises(RuntimeError):
            pipeline.check_validation(duplicate)

    def test_deferred_models_rejected_before_launch(self):
        for model in ('gptoss20b', 'gemma3_12b'):
            with self.assertRaises(RuntimeError), patch.object(pipeline, 'load') as load:
                pipeline.attempt(None, None, self.job('no', model=model), None, None, None)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
