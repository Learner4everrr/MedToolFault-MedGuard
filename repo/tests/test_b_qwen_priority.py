import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.supplement import b_qwen_priority as b
from medtoolbench.supplement.storage_recovery import atomic_json


def job(name='qwen35_9b', stage='test', arm='external_qa', suffix='001'):
    return {'id': stage + '--' + name + '--' + suffix, 'model': name,
            'stage': stage, 'arm': arm, 'episodes': [{'key': suffix}]}


class QwenBPriorityTests(unittest.TestCase):
    def test_only_qwen_external_test(self):
        q = job()
        plan = {'jobs': [q, job('ministral3_8b'), job('gptoss20b'),
                        job('gemma3_12b'), job(arm='decision'), job(stage='qualification')]}
        self.assertEqual(b.selected(plan, 'test'), [q])

    def test_fresh_rejects_any_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / 'job'
            b.require_fresh(folder)
            folder.mkdir()
            b.require_fresh(folder)
            (folder / 'attempt1').mkdir()
            with self.assertRaises(RuntimeError):
                b.require_fresh(folder)

    def test_inventory_ignores_other_model_failures(self):
        with tempfile.TemporaryDirectory() as d:
            cp = Path(d) / 'plan.json'
            q, other = job(), job('ministral3_8b')
            atomic_json(cp.parent / 'jobs' / other['id'] / 'result.json', {'state': 'failed'})
            retained, pending = b.inventory(cp, {'jobs': [q, other]})
            self.assertEqual(retained, [])
            self.assertEqual(pending, [q['id']])

    def test_inventory_preserves_complete_after_audit(self):
        with tempfile.TemporaryDirectory() as d:
            cp, q = Path(d) / 'plan.json', job()
            atomic_json(cp.parent / 'jobs' / q['id'] / 'result.json', {'state': 'complete', 'run': str(cp.parent / 'run')})
            with patch.object(b, 'verify_completion', return_value={'episodes': 1}) as verify:
                retained, pending = b.inventory(cp, {'jobs': [q]})
            verify.assert_called_once()
            self.assertEqual(retained[0]['episodes'], 1)
            self.assertEqual(pending, [])

    def test_inventory_rejects_own_failed_or_unresolved_job(self):
        with tempfile.TemporaryDirectory() as d:
            cp, q = Path(d) / 'plan.json', job()
            atomic_json(cp.parent / 'jobs' / q['id'] / 'result.json', {'state': 'failed'})
            with self.assertRaises(RuntimeError):
                b.inventory(cp, {'jobs': [q]})

    def test_inventory_cannot_start_unqualified_model(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                b.inventory(Path(d) / 'plan.json', {'jobs': [job(stage='qualification')]})

    def test_authorization_forbids_unlisted_model_and_development(self):
        q = job()
        for forbidden in [job('llama31_8b'), job(stage='qualification'), job(suffix='002')]:
            with self.assertRaises(RuntimeError):
                b.authorize({'pending': [q['id']]}, {'jobs': [q, forbidden]}, forbidden)

    def test_authorization_requires_gate_and_freshness(self):
        q = job()
        with patch.object(b, 'model_gate', return_value='failed'), patch.object(b, 'require_fresh') as fresh:
            with self.assertRaises(RuntimeError):
                b.authorize({'pending': [q['id']]}, {'jobs': [q]}, q)
            fresh.assert_not_called()
        with patch.object(b, 'model_gate', return_value='passed'), patch.object(b, 'require_fresh') as fresh:
            b.authorize({'pending': [q['id']]}, {'jobs': [q]}, q)
            fresh.assert_called_once()

    def test_snapshot_excludes_development_and_keeps_partial_separate(self):
        with tempfile.TemporaryDirectory() as d, patch.object(b, 'PLAN', Path(d) / 'plan.json'):
            q, active = job(), job(suffix='002')
            dev = job(stage='qualification')
            for j in (q, dev):
                atomic_json(Path(d) / 'jobs' / j['id'] / 'result.json', {'state': 'complete'})
            atomic_json(Path(d) / 'jobs' / active['id'] / 'attempt1/progress.json', {'completed': 1})
            m = {'formal_episodes': 2, 'retained': [{'stage': 'test', 'episodes': 1}]}
            result = b.snapshot(m, {'jobs': [q, active, dev]}, {'f': ({'name': 'gpu0'}, active)}, [], 0)
            self.assertEqual(result['audited_formal'], 1)
            self.assertEqual(result['active_saved_not_yet_audited'], 1)
            self.assertEqual(result['new_saved_since_launch'], 1)
            self.assertEqual(result['provisional_remaining_hours'], 0)


if __name__ == '__main__':
    unittest.main()
