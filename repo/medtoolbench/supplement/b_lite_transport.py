"""Versioned, model-specific delivery adapters for the remaining B-lite families.

Clinical tools, parser, labels, policy instructions and budgets are not repaired.
The Qwen reference is untouched and remains a separate original-transport stratum.
"""
import copy
import json
import re

from medtoolbench.storage import digest
from medtoolbench.research.models import input_ids
from .runner import SampleModel
from .external_runner import parse, Episode as OriginalEpisode
from .gptoss_model import GptOssModel, MESSAGE, REASONING, FROZEN_DATE
from .gptoss_native import header
from .gptoss_protocol import _object

LLAMA_VERSION = "llama-json-prefill-b-lite-v1"
GPT_VERSION = "harmony-external-b-lite-v1"
LLAMA_INSTRUCTION = r'''
OUTPUT SERIALIZATION REQUIREMENT (format only; all task/tool rules above still apply):
Your entire response must be ONE valid JSON action object. No explanation,
Markdown, comments, or Python outside that object. For a code action, code is a
JSON string: escape line breaks as \n and quotation marks as \". Do not place
literal newlines inside a quoted string. A final response uses the original
{"final":{"rows":[...]}} structure, with actual JSON values in the rows.
The opening brace of your response is already supplied; continue that object.
'''
GPT_INSTRUCTION = '''
NATIVE TOOL TRANSPORT (supersedes only the JSON delivery/feedback convention):
Call the declared functions.metadata, functions.get_resources, or
functions.execute_python_code using exactly one native commentary-channel tool
handoff per turn. The native recipient supplies the tool selector; keep every
other argument and clinical instruction unchanged. Tool feedback is returned as
a native tool message. Finish with the original {"final":{"rows":[...]}} JSON
object in the final channel. Do not emit tool calls as final-channel text.
'''
TOOLS = [{"type": "function", "function": {
    "name": "metadata", "description": "Return the active read-only tool interface metadata.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
    "name": "get_resources", "description": "Retrieve the complete matching FHIR history into retrieved_resources. Returns counts and schema samples.",
    "parameters": {"type": "object", "properties": {
        "resource_type": {"type": "string", "enum": ["Patient", "Encounter", "Observation"]},
        "patient_fhir_id": {"type": "string", "description": "Patient parameter; inspect metadata for active parameter names."},
        "authoritative": {"type": "boolean", "description": "Request source-of-record retrieval."}},
        "required": ["resource_type"], "additionalProperties": True}}},
    {"type": "function", "function": {
    "name": "execute_python_code", "description": "Execute code in the original restricted Python sandbox, using retrieved_resources. No files, network or processes.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                   "required": ["code"], "additionalProperties": False}}}]
NAMES = {t["function"]["name"] for t in TOOLS}


def decode(raw):
    meta = {"version": GPT_VERSION, "kind": "rejected", "frames": [], "arguments_modified": False}
    stream, pos, chosen = "<|start|>assistant" + raw, 0, None
    try:
        for match in MESSAGE.finditer(stream):
            if match.start() != pos or chosen is not None:
                raise ValueError("Malformed boundary or data after terminal action")
            pos = match.end()
            h, body, end = match.groups()
            channel, recipient = header(h)
            if re.search(r"<\|(?:start|channel|message|constrain)\|>", body):
                raise ValueError("Unexpected control token in body")
            meta["frames"].append({"channel": channel, "recipient": recipient, "body": body, "ending": end})
            if recipient is not None:
                name = recipient.removeprefix("functions.")
                if recipient != "functions." + name or name not in NAMES:
                    raise ValueError("Undeclared external function recipient")
                if channel != "commentary" or end != "<|call|>":
                    raise ValueError("Tool call requires commentary and explicit handoff")
                arguments = _object(body)
                if "final" in arguments or "tool" in arguments:
                    raise ValueError("Native arguments must not mix tool/final envelopes")
                chosen = json.dumps({"tool": name, **arguments}, ensure_ascii=False)
                parse(chosen)  # same clinical/action parser, no value repair
                meta.update(kind="tool", recipient=recipient, name=name, arguments=arguments)
            elif channel == "final":
                if end != "<|return|>" or set(_object(body)) != {"final"}:
                    raise ValueError("Final requires exactly the original final envelope and terminal token")
                parse(body)
                chosen = body
                meta.update(kind="final")
            elif end != "<|end|>":
                raise ValueError("Nonterminal content is not an action")
            if end in ("<|call|>", "<|return|>", "<|endoftext|>") and pos != len(stream):
                raise ValueError("Trailing data after terminal")
        if pos != len(stream) or chosen is None:
            raise ValueError("No complete action")
        return chosen, "native_" + meta["kind"], meta
    except (ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
        meta.update(kind="rejected", error=str(error))
        return "", "protocol_rejected", meta


def native_messages(messages):
    if len(messages) < 2 or [m["role"] for m in messages[:2]] != ["system", "user"]:
        raise ValueError("Expected original public prompt prefix")
    result = copy.deepcopy(messages[:2])
    result[0]["content"] += GPT_INSTRUCTION
    waiting = None
    for message in messages[2:]:
        if message["role"] == "assistant":
            if waiting:
                raise ValueError("Missing native tool feedback")
            native = message.get("b_lite_native")
            if not native or native.get("version") != GPT_VERSION:
                raise ValueError("Unversioned native assistant history")
            if native["kind"] == "tool":
                reasoning = "\n".join(f["body"] for f in native["frames"] if f["recipient"] is None)
                result.append({"role": "assistant", "thinking": reasoning, "tool_calls": [
                    {"type": "function", "function": {"name": native["name"], "arguments": copy.deepcopy(native["arguments"])}}]})
                waiting = native["name"]
            else:
                result.append({"role": "assistant", "content": message["content"]})
        elif message["role"] == "user":
            if waiting:
                result.append({"role": "tool", "name": waiting, "content": json.loads(message["content"])})
                waiting = None
            else:
                result.append(copy.deepcopy(message))
        else:
            raise ValueError("Unexpected canonical history role")
    if waiting:
        raise ValueError("Missing tool feedback")
    return result


def render(tokenizer, messages, kind, template=None):
    if kind == "gpt_oss":
        return input_ids(tokenizer.apply_chat_template(native_messages(messages), chat_template=template,
            tools=TOOLS, reasoning_effort=REASONING, tokenize=True, return_dict=False, add_generation_prompt=True))
    adapted = copy.deepcopy(messages)
    if kind == "llama":
        adapted[0]["content"] += LLAMA_INSTRUCTION
    ids = input_ids(tokenizer.apply_chat_template(adapted, tokenize=True, return_dict=False,
                      add_generation_prompt=True, enable_thinking=False))
    return ids + tokenizer.encode("{", add_special_tokens=False) if kind == "llama" else ids


class JsonModel(SampleModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.kind not in {"llama", "mistral3"}:
            raise RuntimeError("This continuation cannot rerun Qwen or substitute another model")
        self.metadata["b_lite_transport"] = LLAMA_VERSION if self.kind == "llama" else "original-json"
        self.metadata["output_parser_unchanged"] = True

    def prompt(self, messages):
        return render(self.tokenizer, messages, self.kind)

    def generate(self, prompts, max_new_tokens):
        rows = super().generate(prompts, max_new_tokens)
        for prompt, row in zip(prompts, rows):
            row["raw_model_text"] = row["text"]
            row["input_prompt_hash"] = digest(prompt)
            row["prompt_prefill"] = "{" if self.kind == "llama" else ""
            row["text"] = row["prompt_prefill"] + row["raw_model_text"]
        return rows


class NativeExternalModel(GptOssModel):
    def __init__(self, *args, **kwargs):
        from .gptoss_compile import install
        repair = install()
        super().__init__(*args, **kwargs)
        self.metadata.update(b_lite_transport=GPT_VERSION, tools=TOOLS,
                             compiler_capacity_repair=repair, argument_repair=False)

    def prompt(self, messages):
        return render(self.tokenizer, messages, self.kind, self.template)

    def generate_sample(self, prompts, limit, temperature):
        rows = super().generate_sample(prompts, limit, temperature)
        for prompt, row in zip(prompts, rows):
            if "b_lite_native" not in row:
                row["text"], row["harmony_status"], row["b_lite_native"] = decode(row["raw_harmony"])
            row["input_prompt_hash"] = digest(prompt)
        return rows


class NativeEpisode(OriginalEpisode):
    def advance(self, generation):
        text, status, meta = decode(generation["raw_harmony"])
        if (text, status, meta) != (generation["text"], generation["harmony_status"], generation["b_lite_native"]):
            raise RuntimeError("Raw native output/metadata mismatch")
        index = len(self.messages)
        super().advance(generation)
        self.messages[index]["b_lite_native"] = copy.deepcopy(meta)
        if meta["kind"] == "rejected":
            self.messages[-1]["content"] = json.dumps({"status": 400, "error": meta["error"],
                "instruction": "Use one declared native tool handoff, or the original final JSON in the final channel."}, separators=(",", ":"))


def tokenizer_for(system):
    from transformers import AutoTokenizer, MistralCommonBackend
    cls = MistralCommonBackend if system["model_type"] == "mistral3" else AutoTokenizer
    tok = cls.from_pretrained(system["path"], local_files_only=True)
    template = tok.chat_template.replace('strftime_now("%Y-%m-%d")', json.dumps(FROZEN_DATE)) if system["model_type"] == "gpt_oss" else None
    return tok, template


def audit_row(row, tokenizer, kind, template=None):
    history, cursor = copy.deepcopy(row["messages"]), 2
    statuses = []
    for generation in row["generations"]:
        ids = render(tokenizer, history[:cursor], kind, template)
        if len(ids) != generation["input_tokens"] or digest(ids) != generation["input_prompt_hash"]:
            raise RuntimeError("Exact input prompt replay mismatch")
        if kind == "gpt_oss":
            text, status, meta = decode(generation["raw_harmony"])
            if text != generation["text"] or status != generation["harmony_status"] or digest(meta) != digest(generation["b_lite_native"]):
                raise RuntimeError("Raw Harmony replay mismatch")
            expected = {"role": "assistant", "content": text, "b_lite_native": meta}
        else:
            prefix = "{" if kind == "llama" else ""
            if generation["prompt_prefill"] != prefix or generation["text"] != prefix + generation["raw_model_text"]:
                raise RuntimeError("JSON prefill/raw output reconstruction mismatch")
            expected = {"role": "assistant", "content": generation["text"]}
            try:
                action = parse(generation["text"])
                status = "final" if "final" in action else "tool"
            except (ValueError, TypeError, KeyError, IndexError, RecursionError):
                status = "format_error"
        if cursor >= len(history) or digest(history[cursor]) != digest(expected):
            raise RuntimeError("Assistant history identity mismatch")
        # Recover raw argument insertion order after gzip's sorted JSON writer.
        history[cursor] = expected
        cursor += 1
        if cursor < len(history) and history[cursor]["role"] == "user":
            cursor += 1
        statuses.append(status)
    if cursor != len(history):
        raise RuntimeError("Unexpected conversation tail")
    return statuses
