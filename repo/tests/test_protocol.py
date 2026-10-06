import unittest
from medtoolbench.run_local import parse_action


class ProtocolTests(unittest.TestCase):
    def test_only_first_action_dispatched(self):
        result = parse_action('{"method":"GET","path":"/metadata"}\n{"final":{"action":"done"}}')
        self.assertEqual(result, {"method": "GET", "path": "/metadata"})

    def test_truncated_json_rejected(self):
        with self.assertRaises(ValueError):
            parse_action('{"method":"POST","body":{')

    def test_final_cannot_include_a_write(self):
        with self.assertRaises(ValueError):
            parse_action('{"final":{},"method":"POST","path":"/Observation"}')

    def test_plain_answer_not_credited(self):
        with self.assertRaises(ValueError):
            parse_action('I placed the order')

    def test_equivalent_terminal_syntax(self):
        self.assertEqual(parse_action('{"action":"none"}'), parse_action('{"final":{"action":"none"}}'))
        self.assertEqual(parse_action('{"value":3.2,"unit":"mmol/L"}')["final"]["value"], 3.2)
