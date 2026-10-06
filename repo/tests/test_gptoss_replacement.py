import unittest
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch
from medtoolbench.supplement.gptoss_model import extract_final
from medtoolbench.supplement.gptoss_pipeline import eligible_jobs, replacement_plan, prepare
from medtoolbench.supplement.storage_recovery import atomic_json
from medtoolbench.supplement.gptoss_mask import decode_safe_builder


class DecodeMaskTests(unittest.TestCase):
    def test_decode_preserves_every_argument_except_compile(self):
        calls = []
        marker = object()
        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return marker
        options = dict(mask_mod=object(), B=2, H=None, Q_LEN=1, KV_LEN=128,
                       device='cuda:0', _compile=True)
        self.assertIs(decode_safe_builder(original)(**options), marker)
        self.assertEqual(calls, [((), {**options, '_compile': False})])
        self.assertTrue(options['_compile'])

    def test_prefill_unmodified(self):
        original = lambda *args, **kwargs: (args, kwargs)
        options = dict(Q_LEN=12288, KV_LEN=12288, _compile=True)
        self.assertEqual(decode_safe_builder(original)(**options), ((), options))

    def test_positional_compile_argument(self):
        original = lambda *args, **kwargs: (args, kwargs)
        args = (object(), 2, None, 1, 128, 'cuda', 128, True)
        self.assertEqual(decode_safe_builder(original)(*args), ((*args[:7], False), {}))


class HarmonyTests(unittest.TestCase):
    def test_final_verbatim(self):
        text = ' {"final":{"value":" CHANNEL "}} '
        self.assertEqual(extract_final('<|channel|>final<|message|>' + text + '<|return|>'), (text, 'final'))

    def test_analysis_not_action(self):
        raw = '<|channel|>analysis<|message|>{"method":"DELETE","path":"Patient/1"}<|end|>'
        raw += '<|start|>assistant<|channel|>final<|message|>{"final":{}}<|return|>'
        self.assertEqual(extract_final(raw), ('{"final":{}}', 'final'))

    def test_incomplete_analysis(self):
        self.assertEqual(extract_final('<|channel|>analysis<|message|>{"final":{}}')[0], '')

    def test_no_channel(self):
        self.assertEqual(extract_final('{"final":{}}')[0], '')

    def test_no_native_tools_advertised(self):
        self.assertEqual(extract_final(' to=python<|channel|>analysis<|message|>print(1)<|call|>')[0], '')

    def test_duplicate_finals_rejected(self):
        raw = '<|channel|>final<|message|>one<|end|><|start|>assistant<|channel|>final<|message|>two<|return|>'
        self.assertEqual(extract_final(raw)[0], '')

    def test_no_post_terminal_execution(self):
        raw = '<|channel|>analysis<|message|>x<|return|><|start|>assistant<|channel|>final<|message|>{}<|return|>'
        self.assertEqual(extract_final(raw)[0], '')

    def test_model_format_failure_not_repaired(self):
        raw = '<|channel|>final<|message|>{"method":<|return|>'
        self.assertEqual(extract_final(raw), ('{"method":', 'final'))

    def test_budget_truncated_final_preserved(self):
        self.assertEqual(extract_final('<|channel|>final<|message|>{"final":{}}'), ('{"final":{}}', 'final'))

    def test_unknown_role(self):
        raw = '<|channel|>analysis<|message|>x<|end|><|start|>user<|channel|>final<|message|>{}<|return|>'
        self.assertEqual(extract_final(raw)[0], '')


class MigrationTests(unittest.TestCase):
    def test_debug_runtime_cannot_authorize_formal_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = root / 'smoke.json'
            atomic_json(smoke, {'all_pass': True, 'diagnostic_only': True})
            with self.assertRaisesRegex(RuntimeError, 'Diagnostic CUDA'):
                prepare(root, smoke, root / 'unit.json')

    def test_retired_model_never_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'plan.json'
            job = {'id': 'qualification--gemma3_12b', 'model': 'gemma3_12b', 'stage': 'qualification'}
            self.assertEqual(eligible_jobs([(cp, {'jobs': [job]})], set(), {}), [])

    def test_completed_active_and_partial_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'plan.json'
            jobs = [{'id': name, 'model': 'gptoss20b', 'stage': 'qualification'} for name in ('complete', 'active', 'partial', 'fresh')]
            atomic_json(cp.parent / 'jobs/complete/result.json', {'state': 'complete', 'accuracy': 0})
            (cp.parent / 'jobs/partial/attempt1').mkdir(parents=True)
            ready = eligible_jobs([(cp, {'jobs': jobs})], {(str(cp), 'active')}, {})
            self.assertEqual([r[4]['id'] for r in ready], ['fresh'])

    def test_own_development_required(self):
        with tempfile.TemporaryDirectory() as tmp, patch('medtoolbench.supplement.gptoss_pipeline.model_gate', return_value='waiting'):
            job = {'id': 'formal', 'model': 'gptoss20b', 'stage': 'test'}
            self.assertEqual(eligible_jobs([(Path(tmp) / 'plan.json', {'jobs': [job]})], set(), {}), [])

    def test_blocked_other_model_does_not_block_healthy(self):
        with tempfile.TemporaryDirectory() as tmp, patch('medtoolbench.supplement.gptoss_pipeline.model_gate', return_value='passed'):
            jobs = [{'id': name, 'model': name, 'stage': 'test'} for name in ('gptoss20b', 'qwen35_9b')]
            ready = eligible_jobs([(Path(tmp) / 'plan.json', {'jobs': jobs})], set(), {'gptoss20b': 'failure'})
            self.assertEqual([r[4]['model'] for r in ready], ['qwen35_9b'])

    def test_replacement_preserves_all_workload_assignments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp, smoke = root / 'parent.json', root / 'smoke.json'
            for file in (cp, smoke, root / 'artifacts.json', root / 'unit_tests_final.json', root / 'mask_equivalence_v1/complete.json'):
                atomic_json(file, {})
            job = {'id': 'x--gemma3_12b', 'model': 'gemma3_12b', 'stage': 'test', 'arm': 'matched',
                'episodes': [{'key': 'task--fault--vanilla'}], 'settings': {'seed': 42, 'max_new_tokens': 4096, 'max_context': 16384}}
            parent = {'plan_hash': 'parent', 'systems': [{'name': 'gemma3_12b', 'batch_size': 2}],
                'jobs': [job], 'source_hashes': {}, 'input_hashes': {}, 'require_oracle': False}
            before = copy.deepcopy(parent)
            new = replacement_plan(parent, cp, root, root, smoke, {})
            self.assertEqual(parent, before)
            self.assertEqual(new['jobs'][0]['episodes'], job['episodes'])
            self.assertEqual(new['jobs'][0]['settings'], job['settings'])
            self.assertEqual(new['jobs'][0]['model'], 'gptoss20b')
            self.assertEqual(new['systems'][0]['batch_size'], 2)
            self.assertTrue(new['gptoss_replacement']['token_budget_includes_reasoning'])


if __name__ == '__main__':
    unittest.main()
