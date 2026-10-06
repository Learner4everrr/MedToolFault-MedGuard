"""Offline, native-MXFP4 GPT-OSS adapter for the unchanged JSON-action protocol.

Reasoning is retained in raw generation traces, never executed as an action.
The full generated stream (including reasoning/control tokens) uses the same
per-turn token budget. No native function schema or extra clinical hints are
introduced. Malformed Harmony remains a model-format error, not a repaired act.
"""
import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path

from medtoolbench.research.models import input_ids

REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
KERNEL_REVISION = "c039a37b84eb546e3e38277e011903560347024b"
FROZEN_DATE = "2026-09-22"
REASONING = "low"
MESSAGE = re.compile(r"<\|start\|>(.*?)<\|message\|>(.*?)(<\|end\|>|<\|return\|>|<\|call\|>|<\|endoftext\|>|$)", re.S)


def extract_final(raw):
    """Accept only the final-channel body; never fish JSON out of analysis."""
    stream = "<|start|>assistant" + raw
    position, final, channels = 0, [], []
    for match in MESSAGE.finditer(stream):
        if match.start() != position:
            return "", "malformed_harmony"
        header, body, ending = match.groups()
        position = match.end()
        fields = header.strip().split("<|channel|>")
        if len(fields) != 2 or fields[0].strip() != "assistant":
            return "", "unsupported_recipient_or_header"
        channel = fields[1].strip()
        channels.append(channel)
        if channel == "final":
            final.append(body)
        elif channel != "analysis":
            return "", "unsupported_channel"
        if ending in {"<|return|>", "<|call|>", "<|endoftext|>"} and position != len(stream):
            return "", "trailing_after_terminal"
        if ending == "<|call|>":
            return "", "unadvertised_native_tool_call"
    if position != len(stream) or len(final) != 1 or channels[-1:] != ["final"]:
        return "", "no_unique_final"
    return final[0], "final"


class GptOssModel:
    def __init__(self, path, thinking=True, seed=20260917):
        cache = Path(__file__).resolve().parents[3] / 'data/gptoss_cache'
        kernel = cache / 'hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots' / KERNEL_REVISION / 'build/torch-cuda'
        if not (kernel / 'metadata.json').exists():
            raise RuntimeError('Pinned, publisher-verified local kernel is missing')
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
            HF_HOME=str(cache), LOCAL_KERNELS='kernels-community/gpt-oss-triton-kernels=' + str(kernel),
            TRITON_CACHE_DIR=str(cache / 'triton'), TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
            TORCHINDUCTOR_COMPILE_THREADS='4')
        import torch
        import transformers
        from .gptoss_mask import install_decode_mask_repair
        from transformers.utils import is_kernels_available, is_triton_available
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
        self.path = Path(path).resolve()
        self.torch, self.kind, self.thinking = torch, "gpt_oss", True
        self.config = json.loads((self.path / "config.json").read_text())
        if self.config["model_type"] != "gpt_oss" or self.config.get("quantization_config", {}).get("quant_method") != "mxfp4":
            raise RuntimeError("Expected the pinned native-MXFP4 GPT-OSS checkpoint")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one allocated CUDA device required; no CPU/API fallback")
        if not is_kernels_available() or not is_triton_available("3.4.0"):
            raise RuntimeError("MXFP4 dependencies invalid; BF16 fallback is forbidden")
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        mask_repair = install_decode_mask_repair()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
        original = self.tokenizer.chat_template
        date_expr = 'strftime_now("%Y-%m-%d")'
        if original.count(date_expr) != 1:
            raise RuntimeError("Unexpected GPT-OSS chat template date expression")
        self.template = original.replace(date_expr, json.dumps(FROZEN_DATE))
        # Catch dependency/architecture fallback BEFORE any weights are loaded.
        validate = Mxfp4HfQuantizer.validate_environment
        def checked_validate(quantizer, *args, **kwargs):
            validate(quantizer, *args, **kwargs)
            if quantizer.quantization_config.dequantize:
                raise RuntimeError("Native MXFP4 unavailable; dequantization is forbidden")
        Mxfp4HfQuantizer.validate_environment = checked_validate
        try:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(str(self.path),
                local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
                attn_implementation="flex_attention", quantization_config=transformers.Mxfp4Config(dequantize=False)).eval()
        finally:
            Mxfp4HfQuantizer.validate_environment = validate
        quantizer = self.model.hf_quantizer
        if quantizer.quantization_config.dequantize:
            raise RuntimeError("Unexpected dequantized model")
        experts = [m for m in self.model.modules() if type(m).__name__ == "Mxfp4GptOssExperts"]
        if len(experts) != self.config["num_hidden_layers"]:
            raise RuntimeError("Not every MoE layer uses MXFP4 experts")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.model.generation_config.pad_token_id
        for key in ("temperature", "top_p", "top_k"):
            setattr(self.model.generation_config, key, None)
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        if self.tokenizer.convert_tokens_to_ids("<|end|>") in self.eos_ids:
            raise RuntimeError("Analysis message boundary cannot terminate generation")
        self.metadata = {"path": str(self.path), "model_type": self.kind, "revision": REVISION,
            "dtype": "native MXFP4 experts / BF16 non-expert weights", "dequantize": False,
            "mx_expert_layers": len(experts), "thinking": True, "reasoning_effort": REASONING,
            "seed": seed, "inference_api": False, "physical_microbatch": 2,
            "gpu": torch.cuda.get_device_name(0), "prompt_date": FROZEN_DATE,
            "attention_implementation": "flex_attention",
            "decode_mask_repair": mask_repair,
            "kernel_revision": KERNEL_REVISION, "local_kernel": str(kernel),
            "chat_template_sha256": hashlib.sha256(self.template.encode()).hexdigest(),
            "protocol": "unchanged JSON action in final channel; all generated tokens budgeted",
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "triton", "kernels")}}

    def prompt(self, messages):
        return input_ids(self.tokenizer.apply_chat_template(messages, chat_template=self.template,
            reasoning_effort=REASONING, tokenize=True, return_dict=False, add_generation_prompt=True))

    def generate(self, prompts, max_new_tokens):
        return self.generate_sample(prompts, max_new_tokens, 0)

    def generate_sample(self, prompts, limit, temperature):
        # Fixed before formal execution, never adjusted based on an episode result.
        if len(prompts) > 2:
            return [row for offset in range(0, len(prompts), 2)
                    for row in self.generate_sample(prompts[offset:offset + 2], limit, temperature)]
        t, tok = self.torch, self.tokenizer
        encoded = tok.pad({"input_ids": prompts}, return_tensors="pt", padding=True,
            return_attention_mask=True).to("cuda:0")
        counts = encoded.attention_mask.sum(dim=1).tolist()
        settings = {"do_sample": bool(temperature)}
        if temperature:
            settings.update(temperature=temperature, top_p=.95, top_k=0)
        start = time.perf_counter()
        with t.inference_mode():
            output = self.model.generate(**encoded, max_new_tokens=limit, **settings,
                pad_token_id=tok.pad_token_id, use_cache=True)
        t.cuda.synchronize()
        elapsed, rows = time.perf_counter() - start, []
        for count, tokens in zip(counts, output[:, encoded.input_ids.shape[1]:].tolist()):
            end = next((i for i, v in enumerate(tokens) if v in self.eos_ids), None)
            used = tokens if end is None else tokens[:end + 1]
            raw = tok.decode(used, skip_special_tokens=False)
            action, status = extract_final(raw)
            rows.append({"text": action, "raw_harmony": raw, "harmony_status": status,
                "input_tokens": int(count), "output_tokens": len(used),
                "generation_limit": end is None and len(used) == limit,
                "batch_size": len(prompts), "batch_seconds_per_item": elapsed / len(prompts)})
        return rows
