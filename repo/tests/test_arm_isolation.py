import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.storage import digest
from medtoolbench.supplement import arm_isolation as arm
from medtoolbench.supplement.storage_recovery import atomic_json


class ArmIsolationTests(unittest.TestCase):
    def job(self, name='j', concurrent=True, model='qwen35_9b'):
        return {'id': name, 'model': model, 'stage': 'qualification',
                'settings': {'stress': {'concurrent': concurrent}}}

    def test_arm_and_deferred_models_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [self.job(), self.job('healthy', False), self.job('gpt', False, 'gptoss20b')]
            rows = arm.eligible([(Path(tmp) / 'plan.json', {'jobs': jobs})], set(), {})
            self.assertEqual([r[4]['id'] for r in rows], ['healthy'])

    def test_direct_dispatch_rejects_isolated_before_launch(self):
        with patch.object(arm, '_ATTEMPT') as launch:
            with self.assertRaises(RuntimeError):
                arm.attempt(None, None, self.job(), None, None, None)
            launch.assert_not_called()

    def test_exact_reviewed_failure_only_is_exempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'plan.json'
            file = cp.parent / 'jobs/j/result.json'
            result = {'state': 'quarantined', 'error': 'reviewed'}
            atomic_json(file, result)
            plans = [(cp, {'jobs': [self.job()]})]
            self.assertEqual(arm.blockers(plans, {str(file): digest(result)}), {})
            self.assertIn('qwen35_9b', arm.blockers(plans, {str(file): 'changed'}))
            self.assertIn('qwen35_9b', arm.blockers(plans, {}))

    def test_other_failure_same_model_still_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'plan.json'
            result = {'state': 'quarantined'}
            approved = cp.parent / 'jobs/j/result.json'
            atomic_json(approved, result)
            atomic_json(cp.parent / 'jobs/other/result.json', result)
            plans = [(cp, {'jobs': [self.job(), self.job('other', False)]})]
            self.assertIn('qwen35_9b', arm.blockers(plans, {str(approved): digest(result)}))

    def test_no_failed_or_started_job_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'plan.json'
            jobs = [self.job(n, False) for n in ('done', 'started', 'fresh')]
            atomic_json(cp.parent / 'jobs/done/result.json', {'state': 'complete'})
            (cp.parent / 'jobs/started/attempt1').mkdir(parents=True)
            rows = arm.eligible([(cp, {'jobs': jobs})], set(), {})
            self.assertEqual([r[4]['id'] for r in rows], ['fresh'])

    def test_development_gate_and_active_ownership_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'plan.json'
            job = self.job(concurrent=False)
            plans = [(cp, {'jobs': [job]})]
            self.assertEqual(arm.eligible(plans, {(str(cp), 'j')}, {}), [])
            job['stage'] = 'test'
            with patch.object(arm.engine, 'model_gate', return_value='waiting'):
                self.assertEqual(arm.eligible(plans, set(), {}), [])


if __name__ == '__main__':
    unittest.main()
