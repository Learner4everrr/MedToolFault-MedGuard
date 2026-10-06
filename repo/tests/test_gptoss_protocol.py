import json
import unittest
from unittest.mock import patch

from medtoolbench.supplement.gptoss_protocol import decode
from medtoolbench.supplement.gptoss_protocol_model import GptOssProtocolModel
from medtoolbench.supplement.gptoss_model import GptOssModel


def frame(recipient, body, channel='commentary', ending='<|call|>', header_first=False):
    header = (' to=' + recipient + '<|channel|>' + channel if header_first else
              '<|channel|>' + channel + ' to=' + recipient)
    return header + '<|constrain|>json<|message|>' + body + ending


class ProtocolTests(unittest.TestCase):
    def test_fhir_verbatim_all_verbs_and_headers(self):
        for verb in ('GET', 'POST', 'PUT', 'DELETE'):
            body = json.dumps({'method': verb, 'path': '/Patient/synthetic'})
            for alias in ('repo_browser.' + verb.lower(), 'repo_browser.send_request',
                          'functions.repo_browser.request'):
                for channel in ('analysis', 'commentary'):
                    for first in (True, False):
                        text, status, meta = decode(frame(alias, body, channel, header_first=first), 'fhir')
                        self.assertEqual(text, body)
                        self.assertEqual(status, 'native_fhir')
                        self.assertEqual(meta['added_fields'], [])

    def test_analysis_not_executed(self):
        prefix = '<|channel|>analysis<|message|>{"method":"DELETE","path":"/Patient/a"}<|end|>'
        raw = prefix + '<|start|>assistant' + frame('repo_browser.get', '{"method":"GET","path":"/metadata"}')
        self.assertEqual(json.loads(decode(raw, 'fhir')[0])['method'], 'GET')
        self.assertEqual(decode(prefix, 'fhir')[0], '')

    def test_external_only_selector_added(self):
        code = 'result = "literal\\n"\nprint(result)'
        text, status, meta = decode(frame('functions.execute_python_code', json.dumps({'code': code})), 'external')
        self.assertEqual(json.loads(text), {'tool': 'execute_python_code', 'code': code})
        self.assertEqual(status, 'native_external')
        self.assertEqual(meta['added_fields'], ['tool'])

    def test_external_existing_selector_verbatim(self):
        body = ' {"tool":"metadata"} '
        self.assertEqual(decode(frame('metadata', body), 'external')[0], body)

    def test_bare_code_header_not_raw_execution(self):
        raw = '<|channel|>commentary to=repo_browser.get code<|message|>{"method":"GET","path":"/metadata"}<|call|>'
        self.assertEqual(decode(raw, 'fhir')[1], 'native_fhir')
        self.assertEqual(decode(raw.replace('{"method":"GET","path":"/metadata"}', 'print(1)'), 'fhir')[0], '')

    def test_final_remains_unrepaired(self):
        for body in (' {"final":{}} ', '{"method":'):
            for ending in ('', '<|return|>'):
                self.assertEqual(decode('<|channel|>final<|message|>' + body + ending, 'fhir')[:2], (body, 'final'))

    def test_missing_message_boundary_not_repaired(self):
        raw = '<|channel|>analysis<|message|>count is one<|end|>'
        raw += '<|start|>assistant<|channel|>final <|constrain|>{"rows":[[1]]}<|return|>'
        self.assertEqual(decode(raw, 'external')[1], 'protocol_rejected')

    def test_invalid_mode_fails_closed(self):
        with self.assertRaises(ValueError):
            decode(frame('metadata', '{}'), 'arbitrary')

    def test_undeclared_mixed_conflicting_invalid_and_duplicate_rejected(self):
        bad = [
            ('fhir', 'python', '{"method":"GET","path":"/metadata"}'),
            ('fhir', 'repo_browser.get', '{"method":"DELETE","path":"/Patient/a"}'),
            ('fhir', 'repo_browser.get', '{"path":"/metadata"}'),
            ('fhir', 'repo_browser.get', '{"method":"GET","path":"/metadata","final":{}}'),
            ('fhir', 'repo_browser.get', '{"method":"GET","method":"DELETE","path":"/metadata"}'),
            ('external', 'execute_python_code', '{"tool":"get_resources","code":"print(1)"}'),
            ('external', 'os.system', '{"code":"print(1)"}'),
            ('external', 'metadata', '{"final":{}}'),
            ('external', 'metadata', '{} {}'),
            ('external', 'metadata', '[]'),
            ('external', 'metadata', '{"value":NaN}'),
            ('external', 'metadata', '{"nested":{"x":1,"x":2}}'),
        ]
        for mode, name, body in bad:
            with self.subTest(mode=mode, name=name, body=body):
                self.assertEqual(decode(frame(name, body), mode)[1], 'protocol_rejected')

    def test_truncation_roles_multiple_actions_and_post_terminal_rejected(self):
        valid = frame('metadata', '{}')
        for raw in (valid.removesuffix('<|call|>'), valid.replace('<|call|>', '<|end|>'),
                    valid + '<|start|>assistant<|channel|>final<|message|>{}<|return|>',
                    valid.replace('commentary', 'final'), valid.replace('to=metadata', 'to=metadata to=metadata'),
                    '<|channel|>analysis<|message|>x<|end|><|start|>user' + valid,
                    '<|channel|>analysis<|message|>{}<|call|>',
                    '<|channel|>final<|message|>{}<|end|><|start|>assistant' + valid):
            with self.subTest(raw=raw):
                self.assertEqual(decode(raw, 'external')[0], '')

    def test_post_decode_does_not_modify_budgets_or_raw_tokens(self):
        row = dict(raw_harmony=frame('metadata', '{}'), text='', harmony_status='unsupported_channel',
                   input_tokens=91, output_tokens=27, generation_limit=False, batch_size=2,
                   batch_seconds_per_item=1.3)
        model = object.__new__(GptOssProtocolModel)
        model.protocol_mode = 'external'
        with patch.object(GptOssModel, 'generate_sample', return_value=[dict(row)]) as call:
            result = model.generate_sample([[1, 2]], 2048, 0)[0]
        call.assert_called_once_with([[1, 2]], 2048, 0)
        for key in set(row) - {'text', 'harmony_status'}:
            self.assertEqual(result[key], row[key])
        self.assertEqual(json.loads(result['text']), {'tool': 'metadata'})

    def test_original_external_dispatcher_and_restricted_code_worker(self):
        from medtoolbench.supplement.external_tools import ExternalTools, Retrieval
        from medtoolbench.supplement.external_runner import parse
        resources = {'Patient': [{'resourceType': 'Patient', 'id': 'synthetic'}],
                     'Encounter': [], 'Observation': [{'resourceType': 'Observation', 'id': 'o1'}]}
        tools = ExternalTools(Retrieval(resources, 'synthetic', 'clean'), 'vanilla')
        def dispatch(name, body):
            text, status, _ = decode(frame(name, json.dumps(body)), 'external')
            self.assertEqual(status, 'native_external')
            return tools.call(parse(text))
        self.assertEqual(dispatch('get_resources', {'resource_type': 'Observation', 'patient_fhir_id': 'synthetic'})['count'], 1)
        self.assertEqual(dispatch('execute_python_code', {'code': 'result = len(retrieved_resources["Observation"])'})['status'], 200)
        self.assertEqual(dispatch('execute_python_code', {'code': 'import os\nresult = os.getcwd()'})['status'], 400)


if __name__ == '__main__':
    unittest.main()
