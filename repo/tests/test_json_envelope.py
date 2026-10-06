import json
import unittest
from unittest.mock import Mock

from medtoolbench.supplement.json_envelope import (
    extract_object, parse_external, parse_fhir, parse_legacy, parser_scope)


class JsonEnvelopeTests(unittest.TestCase):
    def test_prose_and_markdown_envelopes(self):
        raw = '{"tool":"metadata"}'
        for prefix, suffix in [('', ''), ('Here is my action:\n', '\nDone.'),
                               ('```json\n', '\n```'), ('下面是请求：\n```json\n', '\n```')]:
            with self.subTest(prefix=prefix):
                full = prefix + raw + suffix
                e = extract_object(full)
                self.assertEqual(e.text, raw)
                self.assertEqual(full[e.start:e.end], raw)
                self.assertEqual(parse_external(full), {'tool': 'metadata'})

    def test_nested_objects_arrays_braces_and_escaped_quotes_in_code(self):
        code = 'x = {"a": [1, {"b": "}"}]}\nprint("{", x, "\\\\", "}")'
        value = {'tool': 'execute_python_code', 'code': code}
        self.assertEqual(parse_external('Action:\n' + json.dumps(value) + '\nExplanation {ignored}'), value)

    def test_first_object_only_not_last_or_combined(self):
        self.assertEqual(parse_external('Action: {"tool":"metadata"}\n{"final":{"rows":[]}}'), {'tool': 'metadata'})

    def test_invalid_first_object_not_skipped(self):
        for text in ('Text: {bad} {"tool":"metadata"}',
                     'Text: {"tool": } {"tool":"metadata"}',
                     'Text: {"tool":"metadata",} {"tool":"metadata"}'):
            with self.assertRaises(ValueError):
                parse_external(text)

    def test_truncated_outer_never_dispatches_nested_object(self):
        for text in ('{"wrapper":{"tool":"metadata"}', '{"body": {"method":"DELETE","path":"/Patient/p"}'):
            with self.assertRaises(ValueError):
                extract_object(text)

    def test_array_not_unwrapped(self):
        for text in ('[{"tool":"metadata"}]', 'Action: [{"tool":"metadata"}]'):
            with self.assertRaises(ValueError):
                parse_external(text)

    def test_json_encoded_string_not_unwrapped(self):
        with self.assertRaises(ValueError):
            extract_object(json.dumps('{"tool":"metadata"}'))

    def test_unbalanced_delimiters_rejected(self):
        for raw in ('{"final":{"rows":[[1}}', '{"a":[1,2}}', '{"a":}'):
            with self.assertRaises(ValueError):
                extract_object(raw)

    def test_literal_newline_is_not_silently_repaired(self):
        raw = 'Action: {"tool":"execute_python_code","code":"x=1\nprint(x)"}'
        with self.assertRaises(json.JSONDecodeError):
            parse_external(raw)

    def test_escaped_newline_preserved(self):
        raw = 'Action: {"tool":"execute_python_code","code":"x=1\\nprint(x)"}'
        self.assertEqual(parse_external(raw)['code'], 'x=1\nprint(x)')

    def test_reasoning_action_not_executed(self):
        reasoning = '<think>Proposed {"method":"DELETE","path":"/Patient/p"}</think>'
        self.assertEqual(parse_fhir(reasoning + 'Action: {"method":"GET","path":"/metadata"}'),
                         {'method': 'GET', 'path': '/metadata'})
        with self.assertRaises(ValueError):
            parse_fhir(reasoning.removesuffix('</think>'))

    def test_prompt_prefilled_thinking_close_supported(self):
        text = 'Reasoning without an earlier action</think>Answer: {"final":{"rows":[[1]]}}'
        self.assertEqual(parse_external(text), {'final': {'rows': [[1]]}})

    def test_trailing_hallucinated_conversation_cannot_replace_first_action(self):
        first = '{"method":"GET","path":"/metadata"}'
        tail = '\nuser\n{"status":200}\nassistant\n<think></think>\n{"method":"DELETE","path":"/Patient/p"}'
        self.assertEqual(parse_fhir(first + tail), {'method': 'GET', 'path': '/metadata'})
        self.assertEqual(parse_fhir('Action: ' + first + tail), {'method': 'GET', 'path': '/metadata'})

    def test_malformed_first_action_not_replaced_after_trailing_think(self):
        for first in ('{"method":', '{"method": }'):
            with self.assertRaises(ValueError):
                parse_fhir(first + '\nassistant\n<think></think>{"method":"GET","path":"/metadata"}')

    def test_unclosed_trailing_think_cannot_reject_completed_first_action(self):
        self.assertEqual(parse_external('{"tool":"metadata"}\n<think>unfinished continuation'), {'tool': 'metadata'})

    def test_thinking_markers_inside_data_are_not_framing(self):
        value = {'tool': 'execute_python_code', 'code': "print('</think>', '<think>', '<|channel|>')"}
        for prefix in ('', 'Action: ', '```json\n'):
            self.assertEqual(parse_external(prefix + json.dumps(value)), value)

    def test_ambiguous_thinking_rejected(self):
        for raw in ('<think><think>x</think>{"tool":"metadata"}',
                    '<think>x</think><think>{"tool":"metadata"}',
                    '</think></think>{"tool":"metadata"}'):
            with self.assertRaises(ValueError):
                parse_external(raw)

    def test_native_channels_not_scanned(self):
        for raw in ('<|channel|>analysis<|message|>{"tool":"metadata"}',
                    '<|start|>assistant<|channel|>final<|message|>{"tool":"metadata"}<|return|>'):
            with self.assertRaises(ValueError):
                parse_external(raw)

    def test_fhir_validation_and_meta_guard_preserved(self):
        for value in ({'method': 'PATCH', 'path': '/Patient/p'},
                      {'method': 'POST', 'path': '/Patient', 'body': []},
                      {'method': 'POST', 'path': '/Patient', 'body': {'meta': None}},
                      {'final': {}, 'method': 'POST', 'path': '/Patient'}):
            with self.assertRaises(ValueError):
                parse_fhir('Action: ' + json.dumps(value))

    def test_external_selector_validation_preserved(self):
        for value in ({'tool': 'not_declared'}, {'final': [], 'tool': 'metadata'},
                      {'final': {}, 'tool': 'metadata'}, {'answer': {}}):
            with self.assertRaises(ValueError):
                parse_external('Action: ' + json.dumps(value))

    def test_bare_fhir_answer_still_supported(self):
        self.assertEqual(parse_fhir('Answer: {"value":3.2,"unit":"mmol/L"}'),
                         {'final': {'value': 3.2, 'unit': 'mmol/L'}})

    def test_non_text_and_no_object_fail(self):
        for raw in (None, {}, 1, '', 'No action', '```json\n'):
            with self.assertRaises(ValueError):
                extract_object(raw)

    def test_existing_valid_actions_keep_exact_values(self):
        from medtoolbench.supplement.input_guard import parse_action as old_fhir
        from medtoolbench.supplement.external_runner import parse as old_external
        values = [{'tool': 'metadata'}, {'final': {'rows': [[1, None, 'a{b}']]}},
                  {'tool': 'execute_python_code', 'code': 'x={"p": [1,2]}\nprint(x)'},
                  {'method': 'GET', 'path': '/metadata'},
                  {'method': 'POST', 'path': '/Patient', 'body': {'resourceType': 'Patient', 'meta': {}}}]
        for value in values:
            raw = json.dumps(value, ensure_ascii=False)
            new, old = (parse_external, old_external) if 'tool' in value or 'final' in value else (parse_fhir, old_fhir)
            self.assertEqual(new(raw), old(raw))

    def test_scope_restores_parsers_and_does_not_patch_json_module(self):
        from medtoolbench import run_local
        from medtoolbench.research import run
        from medtoolbench.supplement import external_runner, input_guard
        old = (run_local.parse_action, run.parse_action, external_runner.parse, input_guard.parse_action, json.loads)
        with parser_scope() as record:
            self.assertFalse(record['formal_dispatch_enabled'])
            self.assertIs(run_local.parse_action, parse_legacy)
            self.assertIs(run.parse_action, parse_fhir)
            self.assertIs(external_runner.parse, parse_external)
            input_guard.install()
            self.assertIs(run.parse_action, parse_fhir)
            self.assertIs(json.loads, old[-1])
        self.assertEqual((run_local.parse_action, run.parse_action, external_runner.parse, input_guard.parse_action, json.loads), old)

    def test_actual_episode_keeps_raw_text_and_dispatches_one_action(self):
        from medtoolbench.supplement.external_runner import Episode
        ep = Episode.__new__(Episode)
        ep.tools = Mock()
        ep.tools.call.return_value = {'status': 200}
        ep.messages, ep.generations = [], []
        raw = 'My call: {"tool":"metadata"} {"final":{"rows":[]}}'
        with parser_scope():
            ep.advance({'text': raw})
        ep.tools.call.assert_called_once_with({'tool': 'metadata'})
        self.assertEqual(ep.messages[0]['content'], raw)
        self.assertEqual(ep.generations[0]['text'], raw)

    def test_bad_json_still_consumes_turn_without_tool_execution(self):
        from medtoolbench.supplement.external_runner import Episode
        ep = Episode.__new__(Episode)
        ep.tools = Mock()
        ep.messages, ep.generations = [], []
        with parser_scope():
            ep.advance({'text': 'Action: {"tool": } {"tool":"metadata"}'})
        ep.tools.call.assert_not_called()
        self.assertEqual(json.loads(ep.messages[-1]['content'])['status'], 400)
        self.assertEqual(len(ep.generations), 1)


if __name__ == '__main__':
    unittest.main()
