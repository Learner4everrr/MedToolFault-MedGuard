"""Real-GPU development-only equivalence and worst-context memory regression."""
import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path

from medtoolbench.research.cluster_execution import AllocatedGpuLease, validate_worker
from medtoolbench.research.job_state import write_json
from .memory_adapter import install, uninstall
from .memory_execution import load
from .runner import SampleModel, load_plan, public_messages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--worker", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    manifest = load(a.manifest)
    plan = load_plan(Path(manifest["campaigns"][0]))
    row = validate_worker(plan, a.worker)
    if a.out.exists():
        raise RuntimeError("Never overwrite validation evidence")
    a.out.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(row["gpu"])
    started = time.time()
    records = []

    def progress(record):
        records.append(record)
        write_json(a.out / "progress.json", {"completed": len(records), "heartbeat": time.time(),
            "seconds": time.time() - started, "checks": records})
        print(json.dumps(record), flush=True)

    try:
        with AllocatedGpuLease(row):
            import torch
            system = next(s for s in plan["systems"] if s["name"] == "gemma3_12b")
            model = SampleModel(Path(system["path"]), False, 20260917)
            props = torch.cuda.get_device_properties(0)
            if str(props.uuid).removeprefix("GPU-") != row["uuid"].removeprefix("GPU-"):
                raise RuntimeError("Physical GPU mismatch")
            write_json(a.out / "model.json", model.metadata)
            tasks = {t["id"]: t for t in plan["tasks"]}
            jobs = [j for j in plan["jobs"] if j["stage"] == "qualification" and j["model"] == "gemma3_12b"]
            # Public prompts of development assignments, never held-out outcomes.
            inputs = []
            seen = set()
            for job in jobs:
                for e in job["episodes"]:
                    key = (e["task"], e["condition"])
                    if key in seen:
                        continue
                    seen.add(key)
                    inputs.append(model.prompt(public_messages(tasks[e["task"]], e["condition"], job["settings"])))
                    if len(inputs) == 8:
                        break
                if len(inputs) == 8:
                    break
            if len(inputs) != 8:
                raise RuntimeError("Insufficient fixed development prompts")
            # Also exercise long FFN shapes, left padding, and unequal lengths.
            filler = model.tokenizer.encode(" Synthetic infrastructure stress fixture. No clinical decision is requested.", add_special_tokens=False)
            def sized(n):
                return (inputs[0] + filler * (1 + n // len(filler)))[:n]
            pairs = [inputs[i:i + 2] for i in range(0, 8, 2)] + [[sized(2048), sized(1537)], [sized(4096), sized(3073)]]
            original = []
            for i, prompts in enumerate(pairs):
                original.append(model.generate(prompts, 64))
                progress({"kind": "baseline_development_generation", "pair": i, "input_lengths": list(map(len, prompts))})
            adapter = install(model.model, manifest["chunk_tokens"])
            for i, prompts in enumerate(pairs):
                fixed = model.generate(prompts, 64)
                fields = ("text", "input_tokens", "output_tokens", "generation_limit", "batch_size")
                same = all(all(x[k] == y[k] for k in fields) for x, y in zip(original[i], fixed))
                comparison = {"pair": i, "all_equal": same, "baseline": original[i], "tiled": fixed}
                write_json(a.out / ("comparison" + str(i) + ".json"), comparison)
                progress({"kind": "development_generation_comparison", "pair": i, "all_equal": same})
                if not same:
                    raise RuntimeError("Development greedy generations changed; investigate numerical differences before use")

            # Full-sized FFN numerical comparison, with the real BF16 weights.
            module = next(m for m in model.model.modules() if hasattr(m, "_memory_original_forward"))
            x = torch.randn((2, 1025, module.hidden_size), device="cuda", dtype=torch.bfloat16)
            with torch.inference_mode():
                expected = module._memory_original_forward(x)
                actual = module(x)
                max_diff = (expected.float() - actual.float()).abs().max().item()
                equal = torch.equal(expected, actual)
                close = torch.allclose(expected, actual, rtol=.02, atol=.02)
            progress({"kind": "full_weight_ffn_comparison", "shape": list(x.shape), "bitwise_equal": equal,
                      "max_absolute_difference": max_diff, "within_bf16_tolerance": close})
            if not close:
                raise RuntimeError("FFN numerical regression failed")
            del x, expected, actual
            gc.collect()
            torch.cuda.empty_cache()

            # Baseline stress is allowed to OOM; preserve it as evidence. Only
            # synthetic tokens are used; this does not consume benchmark rows.
            uninstall(model.model)
            torch.cuda.reset_peak_memory_stats()
            baseline_oom = False
            try:
                model.generate([sized(14336), sized(14335)], 8)
            except torch.OutOfMemoryError as error:
                baseline_oom = True
                write_json(a.out / "baseline_oom.json", {"error": str(error), "expected_probe_failure": True})
            progress({"kind": "baseline_long_prefill", "input_lengths": [14336, 14335], "oom": baseline_oom,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            gc.collect()
            torch.cuda.empty_cache()
            install(model.model, manifest["chunk_tokens"])
            # Force full decoding in the synthetic probe only, covering both
            # frozen generation budgets at the maximum allowed total context.
            for length, new_tokens in [(14336, 2048), (12288, 4096)]:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                prompts = [sized(length), sized(length - 1)]
                encoded = model.tokenizer.pad({"input_ids": prompts}, return_tensors="pt", padding=True,
                                               return_attention_mask=True).to("cuda:0")
                start = time.monotonic()
                with torch.inference_mode():
                    output = model.model.generate(**encoded, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                        do_sample=False, pad_token_id=model.tokenizer.pad_token_id, use_cache=True)
                torch.cuda.synchronize()
                generated = output.shape[1] - encoded.input_ids.shape[1]
                progress({"kind": "tiled_full_context_and_decode", "batch_size": 2, "input_tokens": length,
                    "forced_new_tokens": new_tokens, "generated_tokens": generated, "all_pass": generated == new_tokens,
                    "seconds": time.monotonic() - start, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "gpu_total_bytes": props.total_memory})
                if generated != new_tokens:
                    raise RuntimeError("Worst-budget synthetic decode incomplete")
                del encoded, output
                gc.collect()
            result = {"all_pass": True, "manifest_hash": manifest["manifest_hash"], "worker": row,
                "adapter": adapter, "development_generation_comparisons": 12, "checks": records,
                "all_benchmark_outcomes_excluded": True, "synthetic_forced_decoding_probe_only": True,
                "scientific_batch_context_generation_budgets_unchanged": True, "time": time.time()}
        write_json(a.out / "complete.json", result)
    except BaseException:
        write_json(a.out / "failed.json", {"error": traceback.format_exc(), "time": time.time(), "checks": records})
        raise


if __name__ == "__main__":
    main()
