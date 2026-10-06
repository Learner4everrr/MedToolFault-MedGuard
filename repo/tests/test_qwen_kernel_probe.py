import unittest
from unittest.mock import patch
from medtoolbench.supplement import qwen_kernel_probe as p
from medtoolbench.supplement import qwen_kernel_window as w


class ProbeTests(unittest.TestCase):
    def test_test_rows_forbidden_before_tokenizing(self):
        with self.assertRaises(RuntimeError):
            p.prompts_from_row({'split': 'test'}, None)

    def test_all_cells_and_original_budget_preserved(self):
        rows = [{'condition': str(c), 'hazard': str(h), 'key': f'{c}-{h}-{n}',
                 'turn': n, 'ids': list(range(n+1))}
                for c in range(3) for h in range(6) for n in range(2)]
        cases = p.choose_cases(rows)
        self.assertEqual(len(cases), 6)
        self.assertTrue(all(len(c['prompts']) == 4 and c['max_new_tokens'] == 2048 for c in cases))
        chosen = [x for c in cases for x in c['prompts']][:18]
        self.assertEqual(len({(x['condition'], x['hazard']) for x in chosen}), 18)
        self.assertTrue(all(len(x['ids']) == 2 for x in chosen))

    def test_missing_cell_rejected(self):
        with self.assertRaises(RuntimeError):
            p.choose_cases([{'condition': 'one', 'hazard': 'clean'}])

    def reports(self):
        output = {'text': 'x', 'input_tokens': 10, 'output_tokens': 1, 'generation_limit': False}
        return ({'cases_hash': 'same', 'cases': [{'name': 'one', 'seconds': 10, 'outputs': [dict(output)]}]},
                {'cases_hash': 'same', 'cases': [{'name': 'one', 'seconds': 5, 'outputs': [dict(output)]}]})

    def test_equal_probe_does_not_authorize_formal(self):
        a, b = self.reports()
        result = p.compare(a, b)
        self.assertTrue(result['all_outputs_identical'])
        self.assertFalse(result['formal_cutover_authorized'])
        self.assertEqual(result['wall_time_ratio_not_equal_work_if_outputs_differ'], 2)

    def test_divergent_greedy_output_blocks_mixing(self):
        a, b = self.reports()
        b['cases'][0]['outputs'][0]['text'] = 'different'
        result = p.compare(a, b)
        self.assertFalse(result['all_outputs_identical'])
        self.assertEqual(result['exact_match'], 0)
        self.assertFalse(result['formal_cutover_authorized'])

    def test_input_mismatch_rejected(self):
        a, b = self.reports()
        b['cases_hash'] = 'other'
        with self.assertRaises(RuntimeError):
            p.compare(a, b)

    def test_missing_case_or_batch_row_rejected(self):
        a, b = self.reports()
        b['cases'][0]['outputs'] = []
        with self.assertRaises(RuntimeError):
            p.compare(a, b)
        b['cases'] = []
        with self.assertRaises(RuntimeError):
            p.compare(a, b)

    def test_pid_reuse_and_zombie_not_signalled(self):
        expected = {'pid': 999, 'start_ticks': '12', 'cmdline': 'probe'}
        for actual in (None, dict(expected, start_ticks='13', state='R'), dict(expected, state='Z')):
            with patch.object(w, 'identity', return_value=actual):
                self.assertFalse(w.same_process(expected))

    def test_only_verified_dispatcher_is_resumed(self):
        with patch.object(w, 'same_process', return_value=False), patch.object(w.os, 'kill') as kill, patch.object(w, 'atomic_json'):
            from pathlib import Path
            w.resume(Path('/tmp'), {'pid': 999})
            kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
