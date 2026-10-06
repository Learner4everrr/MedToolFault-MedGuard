import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from medtoolbench.storage import digest
from medtoolbench.supplement.gptoss_native import (
    decode, header, native_messages, render, advance, RECIPIENT, VERSION, INSTRUCTION)
from medtoolbench.supplement.gptoss_native_model import NativeModel
from medtoolbench.supplement.gptoss_model import GptOssModel, FROZEN_DATE
from medtoolbench.supplement.gptoss_native_pipeline import (
    ROOT, MODEL, COUNTS, CAMPAIGNS, native_plan, scientific_job, candidates, verify_reference)
from medtoolbench.supplement.storage_recovery import atomic_json


def frame(body, recipient=RECIPIENT, ending='<|call|>', channel='commentary'):
    return (' to=' + recipient if recipient else '') + '<|channel|>' + channel + ' json<|message|>' + body + ending


def generation(raw):
    text, status, metadata = decode(raw)
    return dict(raw_harmony=raw, text=text, harmony_status=status, native_transport=metadata)


def history(action=None):
    action = action or {'method': 'GET', 'path': '/Patient/probe'}
    prefix = '<|channel|>analysis<|message|>Need a real read.<|end|><|start|>assistant'
    g = generation(prefix + frame(json.dumps(action)))
    return [{'role': 'system', 'content': 'Original clinical instructions.'}, {'role': 'user', 'content': 'Synthetic prompt.'},
        {'role': 'assistant', 'content': g['text'], 'native_transport': g['native_transport']},
        {'role': 'user', 'content': '{"status":200,"body":{"resourceType":"Patient","id":"probe"}}'}]


class NativeProtocolTests(unittest.TestCase):
    def test_declared_recipient_and_verbatim_payload_all_verbs(self):
        for verb in ('GET', 'POST', 'PUT', 'DELETE'):
            body = ' {"method":' + json.dumps(verb) + ',"path":"/Patient/probe","body":{"active":false}} '
            text, status, m = decode(frame(body))
            self.assertEqual((text, status), (body, 'native_tool'))
            self.assertEqual(m['arguments'], json.loads(body))
            self.assertFalse(m['arguments_modified'])

    def test_supported_header_orders_and_format(self):
        for h in ('assistant to=' + RECIPIENT + '<|channel|>commentary json',
                  'assistant<|channel|>commentary to=' + RECIPIENT + '<|constrain|>json',
                  'assistant<|channel|>commentary to=' + RECIPIENT,
                  'assistant to=' + RECIPIENT + '<|channel|>commentary code'):
            self.assertEqual(header(h), ('commentary', RECIPIENT))

    def test_bad_headers(self):
        for h in ('assistant commentary', 'user<|channel|>commentary',
                  'assistant<|channel|>unknown', 'assistant<|channel|>commentary xml',
                  'assistant<|channel|>commentary json<|constrain|>json',
                  'assistant to=x to=y<|channel|>commentary',
                  'assistant<|channel|>commentary to=', 'assistant<|channel|>final<|constrain|>xml'):
            with self.subTest(header=h), self.assertRaises(ValueError):
                header(h)

    def test_analysis_is_never_an_action(self):
        raw = '<|channel|>analysis<|message|>{"method":"DELETE","path":"/Patient/probe"}<|end|>'
        self.assertEqual(decode(raw)[1], 'protocol_rejected')
        raw += '<|start|>assistant' + frame('{"method":"GET","path":"/Patient/probe"}')
        self.assertEqual(json.loads(decode(raw)[0])['method'], 'GET')

    def test_strict_json_rejects_duplicates_trailing_nonfinite_nonobjects(self):
        for body in ('{} {}', '[]', '{"method":"GET","method":"DELETE"}',
                     '{"body":{"x":1,"x":2}}', '{"body":{"x":NaN}}', '{"final":{}}', '{"code":"print(1)"}'):
            self.assertEqual(decode(frame(body))[1], 'protocol_rejected')

    def test_unknown_recipient_wrong_channel_and_end(self):
        for raw in (frame('{}', 'repo_browser.get'), frame('{}', 'python'), frame('{}', channel='analysis'),
                    frame('{}', ending='<|end|>'), frame('{}', channel='final'), frame('{}', ending='')):
            self.assertEqual(decode(raw)[0], '')

    def test_multiple_terminal_frames_and_trailing_garbage(self):
        call = frame('{"method":"GET","path":"/Patient/probe"}')
        for raw in (call + 'garbage', call + '<|start|>assistant' + call,
                    '<|channel|>analysis<|message|>x<|end|>bad<|start|>assistant' + call,
                    '<|channel|>analysis<|message|>x<|end|><|start|>user' + call):
            self.assertEqual(decode(raw)[1], 'protocol_rejected')

    def test_final_requires_actual_final_envelope(self):
        raw = frame('{"final":{"value":4}}', None, '<|return|>', 'final')
        self.assertEqual(decode(raw)[:2], ('{"final":{"value":4}}', 'native_final'))
        # The frozen clinical parser already accepts bare answer objects.
        self.assertEqual(decode(frame('{}', None, '<|return|>', 'final'))[1], 'native_final')
        for body in ('{"method":"GET","path":"/metadata"}', '{"final":'):
            self.assertEqual(decode(frame(body, None, '<|return|>', 'final'))[1], 'protocol_rejected')
        self.assertEqual(decode(raw.replace('<|return|>', ''))[1], 'protocol_rejected')

    def test_invalid_clinical_arguments_not_inferred_or_repaired(self):
        for action in ({'path': '/Patient/probe'}, {'method': 'WRONG', 'path': '/Patient/probe'},
                       {'method': 'POST', 'path': '/Observation', 'body': 'invalid'},
                       {'method': 'POST', 'path': '/Observation', 'body': {'meta': None}}):
            self.assertEqual(decode(frame(json.dumps(action)))[2]['arguments'], action)
        self.assertEqual(decode(frame('{}'))[1], 'protocol_rejected')

    def test_history_pure_native_tool_feedback_and_reasoning(self):
        original = history(); before = copy.deepcopy(original)
        messages = native_messages(original)
        self.assertEqual(original, before)
        self.assertEqual(messages[0]['content'], before[0]['content'] + INSTRUCTION)
        self.assertEqual(messages[1], before[1])
        self.assertEqual(messages[2]['thinking'], 'Need a real read.')
        self.assertEqual(messages[2]['tool_calls'][0]['function']['arguments'], {'method': 'GET', 'path': '/Patient/probe'})
        self.assertEqual(messages[3], {'role': 'tool', 'name': 'fhir_request', 'content': json.loads(before[3]['content'])})

    def test_invalid_history_fails_closed(self):
        with self.assertRaises(ValueError):
            native_messages(history()[:-1])
        m = history(); m[2].pop('native_transport')
        with self.assertRaises(ValueError):
            native_messages(m)

    def test_two_by_two_serialization_controls(self):
        for schema in (True, False):
            for hist in (True, False):
                m = native_messages(history(), schemas=schema, history=hist)
                self.assertEqual(INSTRUCTION in m[0]['content'], schema)
                self.assertEqual(m[-1]['role'] == 'tool', hist)
                self.assertEqual('tool_calls' in m[-2], hist)

    def test_generation_accounting_untouched(self):
        row = dict(raw_harmony=frame('{"method":"GET","path":"/metadata"}'),
                   text='old', harmony_status='unsupported', input_tokens=2, output_tokens=31,
                   generation_limit=False, batch_size=1, batch_seconds_per_item=2.5)
        model = object.__new__(NativeModel)
        with patch.object(GptOssModel, 'generate_sample', return_value=[dict(row)]):
            result = model.generate_sample([[1, 2]], 2048, 0)[0]
        for key in set(row) - {'text', 'harmony_status'}:
            self.assertEqual(result[key], row[key])
        self.assertEqual(result['native_prompt_hash'], digest([1, 2]))

    def test_original_executor_and_budget_still_used(self):
        from medtoolbench.supplement.transport import CallBudget, FactorClient
        from medtoolbench.supplement.input_guard import install
        install()
        calls = []
        def raw(method, path, body=None, actor='agent'):
            calls.append((method, path, body, actor))
            return {'status': 200, 'body': {'id': 'probe'}}
        e = SimpleNamespace(messages=history()[:2], generations=[], parsed_actions=[], done=False, answer={}, stop='turn_limit')
        budget = CallBudget(raw, 1); e.client = FactorClient(budget, 'vanilla')
        g = generation(frame('{"method":"GET","path":"/Patient/probe"}'))
        advance(e, g); advance(e, g)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(e.messages[-1]['content'])['status'], 429)
        self.assertEqual(budget.used, 1)
        for action in ({'path': '/Patient/probe'}, {'method': 'POST', 'path': '/Observation', 'body': {'meta': None}}):
            advance(e, generation(frame(json.dumps(action))))
            self.assertEqual(json.loads(e.messages[-1]['content'])['status'], 400)
        self.assertEqual(len(calls), 1)

    def test_rejected_protocol_feedback_does_not_execute(self):
        e = SimpleNamespace(messages=history()[:2], generations=[], parsed_actions=[], done=False,
                            client=SimpleNamespace(call=lambda *a: self.fail('Unexpected execution')))
        advance(e, generation(frame('{}', 'unknown')))
        self.assertIn('undeclared recipient', e.messages[-1]['content'])
        native_messages(e.messages)

    def test_record_mismatch_fails_before_execution(self):
        e = SimpleNamespace(messages=history()[:2])
        g = generation(frame('{}')); g['text'] = '{"method":"DELETE"}'
        with self.assertRaises(RuntimeError):
            advance(e, g)
        self.assertEqual(len(e.messages), 2)


class NativeTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / 'data/models/gpt-oss-20b/tokenizer_config.json').is_file():
            raise unittest.SkipTest('Optional pinned GPT-OSS tokenizer is not bundled in the code release')
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(str(ROOT / 'data/models/gpt-oss-20b'), local_files_only=True)
        cls.template = cls.tok.chat_template.replace('strftime_now("%Y-%m-%d")', json.dumps(FROZEN_DATE))

    def test_real_pinned_template(self):
        text = self.tok.decode(render(self.tok, self.template, history()))
        self.assertIn('namespace functions', text)
        self.assertIn('assistant to=functions.fhir_request<|channel|>commentary json<|message|>', text)
        self.assertIn('Need a real read.', text)
        tool_prefix = '<|start|>functions.fhir_request to=assistant<|channel|>commentary<|message|>'
        self.assertIn(tool_prefix + '{', text)
        self.assertNotIn(tool_prefix + '"', text)
        self.assertTrue(text.endswith('<|start|>assistant'))
        self.assertIn(FROZEN_DATE, text)

    def test_every_generation_prompt_replay_and_tamper(self):
        from medtoolbench.research.build import save_gzip
        from medtoolbench.supplement.gptoss_native_runner import audit_native
        messages = history()
        call = generation('<|channel|>analysis<|message|>Need a real read.<|end|><|start|>assistant' +
                          frame('{"method": "GET", "path": "/Patient/probe"}'))
        final = generation(frame('{"final":{}}', None, '<|return|>', 'final'))
        for g, prefix in ((call, messages[:2]), (final, messages)):
            tokens = render(self.tok, self.template, prefix)
            g.update(input_tokens=len(tokens), native_prompt_hash=digest(tokens))
        messages.append({'role': 'assistant', 'content': final['text'], 'native_transport': final['native_transport']})
        row = {'messages': messages, 'generations': [call, final]}
        plan = {'systems': [{'path': str(ROOT / 'data/models/gpt-oss-20b')}], 'plan_hash': 'synthetic'}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); (out / 'episodes').mkdir()
            file = out / 'episodes/probe.json.gz'
            save_gzip(file, row)
            self.assertTrue(audit_native(out, plan)['all_pass'])
            row['generations'][0]['native_prompt_hash'] = 'tampered'
            save_gzip(file, row)
            with self.assertRaises(RuntimeError):
                audit_native(out, plan)


class NativePipelineTests(unittest.TestCase):
    def test_complete_acd_scientific_assignments_unchanged(self):
        missing = [name for name in CAMPAIGNS.values() if not (ROOT / 'data' / name / 'plan.json').is_file()]
        if missing:
            self.skipTest('Full campaign fixtures obtained separately: ' + ', '.join(missing))
        for letter, name in CAMPAIGNS.items():
            cp = ROOT / 'data' / name / 'plan.json'
            parent = json.loads(cp.read_text())
            plan = native_plan(parent, cp, [], {})
            expected = [j for j in parent['jobs'] if j['model'] == 'qwen35_9b']
            self.assertEqual([scientific_job(j) for j in expected], [scientific_job(j) for j in plan['jobs']])
            self.assertEqual({j['model'] for j in plan['jobs']}, {MODEL})
            totals = tuple(sum(len(j['episodes']) for j in plan['jobs'] if j['stage'] == s) for s in ('qualification', 'test'))
            self.assertEqual(totals, COUNTS[letter])
            self.assertFalse(plan['native_amendment']['clinical_accuracy_gate'])
            self.assertEqual(parent, json.loads(cp.read_text()))

    def test_b_is_not_authorized(self):
        with self.assertRaises(RuntimeError):
            native_plan({}, ROOT / 'data/supplement_b_v3/plan.json', [], {})

    def test_global_development_gate_and_no_retries(self):
        jobs = [dict(id='dev', model=MODEL, stage='qualification'), dict(id='formal', model=MODEL, stage='test')]
        with tempfile.TemporaryDirectory() as directory:
            cp = Path(directory) / 'plan.json'; plans = [(cp, {'jobs': jobs})]
            self.assertEqual([r[-1]['id'] for r in candidates(plans, set(), False)], ['dev'])
            self.assertEqual(len(candidates(plans, set(), True)), 2)
            (cp.parent / 'jobs/dev/attempt1').mkdir(parents=True)
            self.assertEqual(candidates(plans, {(str(cp), 'formal')}, True), [])

    def test_analysis_references_resolve_original_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'jobs/one/result.json'; cp = root / 'plan.json'
            job = {'id': 'one', 'model': MODEL, 'episodes': [], 'stage': 'test'}
            original = {'job': 'one', 'state': 'complete', 'run': '/synthetic/unchanged'}
            atomic_json(source, original)
            parent = {'plan_hash': 'original', 'jobs': [job]}
            view = {'source_plans': {str(cp): 'original'}}
            reference = {**original, 'analysis_reference_only': True, 'source_result': str(source),
                'source_result_hash': digest(original), 'source_plan_hash': 'original'}
            with patch('medtoolbench.supplement.gptoss_native_pipeline.load_plan', return_value=parent), \
                 patch('medtoolbench.supplement.gptoss_native_pipeline.verify_completion', return_value='ok') as verify:
                self.assertEqual(verify_reference(reference, view, job), 'ok')
                verify.assert_called_once_with(Path('/synthetic/unchanged'), parent, job)
                reference['source_result_hash'] = 'tampered'
                with self.assertRaises(RuntimeError):
                    verify_reference(reference, view, job)

    def test_positive_probe_cannot_accept_only_errors_or_guessed_answers(self):
        spec = importlib.util.spec_from_file_location('native_probe_test', ROOT / 'repo/scripts/gptoss_native_probe.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        e = SimpleNamespace(done=True, answer={'family': 'test', 'value': 8, 'id': 'created'}, trace=[])
        self.assertFalse(module.positive(e, 'test', 7, []))
        resource = {'resourceType': 'Observation', 'id': 'created', 'valueQuantity': {'value': 8},
                    'subject': {'reference': 'Patient/native-probe'}}
        e.trace = [{'method': method, 'path': path, 'response': {'status': 200, 'body': resource}} for method, path in
                   [('GET', '/Patient/native-probe'), ('GET', '/Observation/native-source'),
                    ('POST', '/Observation'), ('GET', '/Observation/created')]]
        self.assertTrue(module.positive(e, 'test', 7, [resource]))
        e.trace[2]['response']['status'] = 400
        self.assertFalse(module.positive(e, 'test', 7, [resource]))


if __name__ == '__main__':
    unittest.main()
