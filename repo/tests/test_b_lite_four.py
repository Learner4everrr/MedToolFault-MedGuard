import copy
import json
import unittest
from unittest.mock import patch

from medtoolbench.storage import digest
from medtoolbench.supplement import b_lite_transport as transport
from medtoolbench.supplement.b_lite_four import clone_jobs, MODELS
from medtoolbench.supplement.b_lite import SETTINGS
from medtoolbench.supplement.external_runner import parse


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        # Real chat templates address message fields by name, but iterate raw
        # nested tool-argument mappings in insertion order.
        messages = [{k: m[k] for k in sorted(m)} for m in messages]
        return list(json.dumps([messages, kwargs], ensure_ascii=False).encode())

    def encode(self, text, **kwargs):
        return list(text.encode())


PREFIX = [{"role": "system", "content": "Original public instructions"},
          {"role": "user", "content": "Original question; no gold labels"}]


def native_generation(raw, history):
    text, status, meta = transport.decode(raw)
    ids = transport.render(Tokenizer(), history, "gpt_oss")
    return {"raw_harmony": raw, "text": text, "harmony_status": status,
            "b_lite_native": meta, "input_tokens": len(ids), "input_prompt_hash": digest(ids)}


class FourGpuTransportTests(unittest.TestCase):
    def test_original_parser_stays_strict(self):
        with self.assertRaises(ValueError):
            parse('Here is the result: {"final":{"rows":[]}}')

    def test_llama_prefill_is_input_not_output_repair(self):
        original = copy.deepcopy(PREFIX)
        ids = transport.render(Tokenizer(), original, "llama")
        self.assertEqual(ids[-1], ord("{"))
        self.assertEqual(original, PREFIX)
        raw = '"tool":"metadata"}'
        row = {"messages": PREFIX + [{"role": "assistant", "content": "{" + raw},
                {"role": "user", "content": '{"status":200}'}], "generations": [{
                    "input_tokens": len(ids), "input_prompt_hash": digest(ids),
                    "raw_model_text": raw, "prompt_prefill": "{", "text": "{" + raw}]}
        self.assertEqual(transport.audit_row(row, Tokenizer(), "llama"), ["tool"])
        row["generations"][0]["raw_model_text"] = '"tool":"get_resources"}'
        with self.assertRaises(RuntimeError):
            transport.audit_row(row, Tokenizer(), "llama")

    def test_ministral_prompt_is_original(self):
        expected = Tokenizer().apply_chat_template(PREFIX, tokenize=True, return_dict=False,
                    add_generation_prompt=True, enable_thinking=False)
        self.assertEqual(transport.render(Tokenizer(), PREFIX, "mistral3"), expected)

    def test_three_original_tools_only(self):
        self.assertEqual(transport.NAMES, {"metadata", "get_resources", "execute_python_code"})
        params = transport.TOOLS[1]["function"]["parameters"]
        self.assertNotIn("subject_fhir_id", params["properties"])
        self.assertTrue(params["additionalProperties"])

    def test_native_tool_arguments_not_changed(self):
        raw = '<|channel|>analysis<|message|>Ignore this {"final":{}}<|end|><|start|>assistant to=functions.get_resources<|channel|>commentary<|message|>{"resource_type":"Observation","subject_fhir_id":"p","authoritative":true}<|call|>'
        text, status, meta = transport.decode(raw)
        self.assertEqual(status, "native_tool")
        self.assertEqual(parse(text), {"tool": "get_resources", "resource_type": "Observation",
                         "subject_fhir_id": "p", "authoritative": True})
        self.assertFalse(meta["arguments_modified"])

    def test_native_final(self):
        text, status, _ = transport.decode('<|channel|>final<|message|>{"final":{"rows":[[3]]}}<|return|>')
        self.assertEqual(status, "native_final")
        self.assertEqual(parse(text), {"final": {"rows": [[3]]}})

    def test_fail_closed(self):
        bad = [
            '<|channel|>analysis<|message|>{"final":{"rows":[]}}<|end|>',
            ' to=functions.metadata<|channel|>commentary<|message|>{}<|end|>',
            ' to=functions.metadata<|channel|>final<|message|>{}<|call|>',
            ' to=functions.unknown<|channel|>commentary<|message|>{}<|call|>',
            ' to=functions.metadata<|channel|>commentary<|message|>{"final":{}}<|call|>',
            ' to=functions.metadata<|channel|>commentary<|message|>{"x":1,"x":2}<|call|>',
            ' to=functions.metadata<|channel|>commentary<|message|>{"x":NaN}<|call|>',
            ' to=functions.metadata<|channel|>commentary<|message|>{}{}<|call|>',
            '<|channel|>final<|message|>{"final":{},"tool":"metadata"}<|return|>',
            '<|channel|>final<|message|>{"final":{"rows":[]}}<|return|>junk',
        ]
        for raw in bad:
            with self.subTest(raw=raw):
                self.assertEqual(transport.decode(raw)[:2], ("", "protocol_rejected"))

    def test_native_roundtrip_preserves_raw_argument_order(self):
        history = copy.deepcopy(PREFIX)
        first = native_generation(' to=functions.get_resources<|channel|>commentary<|message|>{"resource_type":"Observation","patient_fhir_id":"p","authoritative":true}<|call|>', history)
        history.extend([{"role": "assistant", "content": first["text"], "b_lite_native": first["b_lite_native"]},
                        {"role": "user", "content": '{"status":200,"count":1}'}])
        second = native_generation('<|channel|>final<|message|>{"final":{"rows":[[1]]}}<|return|>', history)
        history.append({"role": "assistant", "content": second["text"], "b_lite_native": second["b_lite_native"]})
        row = json.loads(json.dumps({"messages": history, "generations": [first, second]}, sort_keys=True))
        self.assertEqual(transport.audit_row(row, Tokenizer(), "gpt_oss"), ["native_tool", "native_final"])
        row["generations"][1]["input_prompt_hash"] = "tampered"
        with self.assertRaises(RuntimeError):
            transport.audit_row(row, Tokenizer(), "gpt_oss")

    def test_native_feedback_uses_tool_role(self):
        gen = native_generation(' to=functions.metadata<|channel|>commentary<|message|>{}<|call|>', PREFIX)
        messages = PREFIX + [{"role": "assistant", "content": gen["text"], "b_lite_native": gen["b_lite_native"]},
                             {"role": "user", "content": '{"status":200}'}]
        rendered = transport.native_messages(messages)
        self.assertEqual(rendered[-1], {"role": "tool", "name": "metadata", "content": {"status": 200}})
        with self.assertRaises(ValueError):
            transport.native_messages(messages[:-1])

    def test_clone_has_same_cells_no_qwen(self):
        jobs = []
        for stage in ("qualification", "test"):
            jobs.append({"id": stage + "--qwen35_9b--000", "model": "qwen35_9b", "stage": stage,
                         "episodes": [{"key": "same-original-cell"}], "settings": SETTINGS})
        result = clone_jobs({"jobs": jobs})
        self.assertEqual(len(result), 2 * len(MODELS))
        self.assertEqual(MODELS, ("llama31_8b",))
        self.assertEqual({j["model"] for j in result}, set(MODELS))
        self.assertTrue(all(j["episodes"] == jobs[0]["episodes"] and j["settings"] == SETTINGS for j in result))


if __name__ == "__main__":
    unittest.main()
