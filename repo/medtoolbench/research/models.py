"""In-process offline Transformers adapters, no inference server/API."""
import hashlib
import json
import os
import time
from pathlib import Path
from collections.abc import Mapping


def input_ids(encoded):
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, list) or any(not isinstance(t, int) for t in encoded):
        raise ValueError("Expected a single unbatched sequence of chat token IDs")
    return encoded


class LocalModel:
    def __init__(self, path, thinking=False, seed=20260911):
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        import torch
        import transformers
        self.torch, self.thinking = torch, thinking
        self.path = Path(path).resolve()
        self.config = json.loads((self.path / "config.json").read_text())
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no API fallback")
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            for name in set(json.loads(index.read_text())["weight_map"].values()):
                if not (self.path / name).is_file():
                    raise FileNotFoundError(f"Incomplete local snapshot: {name}")
        self.kind = self.config["model_type"]
        if self.kind == "mistral3":
            self.tokenizer = transformers.MistralCommonBackend.from_pretrained(str(self.path), local_files_only=True)
            loader = transformers.Mistral3ForConditionalGeneration
        elif self.kind == "qwen3_5":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
            loader = transformers.Qwen3_5ForConditionalGeneration
        elif self.kind == "gemma3":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
            loader = transformers.Gemma3ForConditionalGeneration
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
            loader = transformers.AutoModelForCausalLM
        kwargs = dict(local_files_only=True, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
        self.dequantized = self.config.get("quantization_config", {}).get("quant_method") == "fp8"
        if self.dequantized:
            kwargs["quantization_config"] = transformers.FineGrainedFP8Config(dequantize=True)
        self.model = loader.from_pretrained(str(self.path), **kwargs).eval()
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        for key in ("temperature", "top_p", "top_k"):
            setattr(self.model.generation_config, key, None)
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        self.metadata = {"path": str(self.path), "model_type": self.kind,
            "model_config_sha256": hashlib.sha256((self.path / "config.json").read_bytes()).hexdigest(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16", "fp8_dequantized": self.dequantized,
            "thinking": thinking, "do_sample": False, "seed": seed, "inference_api": False,
            "download_manifest": json.loads((self.path / "DOWNLOAD_MANIFEST.json").read_text())
                                  if (self.path / "DOWNLOAD_MANIFEST.json").exists() else None}

    def prompt(self, messages):
        if self.kind == "gemma3":
            # Gemma's documented template has no separate system role. Preserve
            # every instruction verbatim by concatenating it to the first user turn.
            messages = [dict(m) for m in messages]
            if messages[0]["role"] == "system":
                system = messages.pop(0)["content"]
                messages[0]["content"] = system + "\n\n" + messages[0]["content"]
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                    add_generation_prompt=True, enable_thinking=self.thinking)
        return input_ids(encoded)

    def generate(self, prompts, max_new_tokens):
        t, tok = self.torch, self.tokenizer
        # Native chat-template token IDs avoid lossy decode/re-encode of control
        # tokens, especially with the official Mistral tokenizer backend.
        encoded = tok.pad({"input_ids": prompts}, return_tensors="pt", padding=True,
                          return_attention_mask=True).to("cuda:0")
        input_counts = encoded.attention_mask.sum(dim=1).tolist()
        started = time.perf_counter()
        with t.inference_mode():
            generated = self.model.generate(**encoded, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        t.cuda.synchronize()
        elapsed = time.perf_counter() - started
        rows = []
        for count, output in zip(input_counts, generated[:, encoded.input_ids.shape[1]:].tolist()):
            end = next((i for i, token in enumerate(output) if token in self.eos_ids), None)
            used = output if end is None else output[:end+1]
            rows.append({"text": tok.decode(used, skip_special_tokens=True), "input_tokens": int(count),
                         "output_tokens": len(used), "generation_limit": end is None and len(used) == max_new_tokens,
                         "batch_size": len(prompts), "batch_seconds_per_item": elapsed / len(prompts)})
        return rows


def smoke():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    model = LocalModel(a.model)
    prompts = [model.prompt([{"role": "user", "content": 'Reply with exactly {"ready":true} and nothing else.'}])]
    rows = model.generate(prompts, 32)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"model": model.metadata, "generations": rows}, indent=2) + "\n")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    smoke()
