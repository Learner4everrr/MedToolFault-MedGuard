import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.supplement import acd_order as order
from medtoolbench.supplement.storage_recovery import read


class ACDOrderTests(unittest.TestCase):
    def plans(self):
        return [(Path('/test') / name / 'plan.json', {'jobs': []}) for name in
                ('supplement_a_v3', 'supplement_b_v2', 'supplement_c_v4', 'supplement_d_v3')]

    def test_only_acd_allowed(self):
        self.assertEqual([order.is_acd(p) for p, _ in self.plans()], [True, False, True, True])
        self.assertFalse(order.is_acd('/test/other/plan.json'))

    def test_b_never_evaluated_for_dispatch(self):
        with patch.object(order, '_ELIGIBLE', return_value=[]) as call:
            order.eligible(None, self.plans(), {}, set(), {})
            self.assertEqual([p.parent.name for p, _ in call.call_args.args[1]],
                             ['supplement_a_v3', 'supplement_c_v4', 'supplement_d_v3'])

    def test_b_pending_does_not_block_acd_finalization(self):
        with patch.object(order, '_INVENTORY', return_value=([], {}, 0)) as call:
            self.assertEqual(order.inventory(None, self.plans(), {}), ([], {}, 0))
            self.assertEqual(len(call.call_args.args[1]), 3)

    def test_repair_overlay_references_original_control(self):
        cp = Path('/test/supplement_a_v3/plan.json')
        key = str(cp.resolve()) + '::concurrent'
        m = {'scheduling_parent': '/original/control/manifest.json', 'repair_jobs': {key: {}}}
        self.assertEqual(order.folder_for(Path('/new'), cp, {'id': 'concurrent'}, m),
                         Path('/original/control/repair_jobs/concurrent'))
        self.assertEqual(order.folder_for(Path('/new'), cp, {'id': 'normal'}, m), cp.parent / 'jobs/normal')

    def test_completed_receipt_never_claims_four_models_or_whole_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = Path(tmp)
            m = {'allowed_models': ['qwen35_9b'], 'manifest_hash': 'test'}
            plans = [(control / 'supplement_b_v2/plan.json', {'jobs': [
                {'id': 'b', 'model': 'qwen35_9b', 'episodes': [1, 2]}]})]
            with patch.object(order, 'folder_for', return_value=control / 'b'):
                order.finalize(control, plans, m)
            proof = read(control / 'complete.json')
            self.assertFalse(proof['entire_project_complete'])
            self.assertFalse(proof['four_model_acd_complete'])
            self.assertFalse(proof['automatic_b_resume'])
            self.assertEqual(proof['deferred_b'][0]['episodes'], 2)

    def test_missing_acd_result_prevents_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = Path(tmp)
            m = {'allowed_models': ['qwen35_9b'], 'manifest_hash': 'test'}
            plans = [(control / 'supplement_a_v3/plan.json', {'jobs': [
                {'id': 'a', 'model': 'qwen35_9b', 'episodes': [1]}]})]
            with patch.object(order, 'folder_for', return_value=control / 'missing'):
                with self.assertRaisesRegex(RuntimeError, 'Unresolved ACD'):
                    order.finalize(control, plans, m)


if __name__ == '__main__':
    unittest.main()
