"""Separate real-GPU comparison and sustained mixed-length memory probes."""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

from medtoolbench.research.cluster_execution import AllocatedGpuLease, validate_worker
from medtoolbench.research.job_state import write_json
from .isolated_manifest import load, require_cpu_validation
from .memory_adapter import install
from .runner import SampleModel, load_plan, public_messages
from .serial_model import SerialGemma


def public_inputs(model, plan):
    tasks = {t["id"]: t for t in plan["tasks"]}
    inputs, seen = [], set()
    for job in plan["jobs"]:
        if job["stage"] != "qualification" or job["model"] != "gemma3_12b":
            continue
        for entry in job["episodes"]:
            key = entry["task"], entry["condition"]
            if key not in seen:
                seen.add(key)
                inputs.append(model.prompt(public_messages(tasks[entry["task"]], entry["condition"], job["settings"])))
            if len(inputs) == 8:
                return inputs
    raise RuntimeError("Insufficient development prompts")


def main():
    p = argparse.ArgumentParser()
    for field in ("manifest", "out"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--worker", required=True)
    p.add_argument("--mode", choices=("compare", "stress"), required=True)
    a = p.parse_args()
    manifest = load(a.manifest)
    require_cpu_validation(a.manifest.parent, manifest)
    plan = load_plan(Path(manifest["campaigns"][0]))
    worker = validate_worker(plan, a.worker)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker["gpu"])
    if a.out.exists():
        raise RuntimeError("Never overwrite a preflight")
    a.out.mkdir(parents=True)
    checks, start = [], time.monotonic()

    def progress(row):
        checks.append(row)
        write_json(a.out / "progress.json", {"completed": len(checks), "checks": checks,
            "seconds": time.monotonic() - start, "heartbeat": time.time()})
        print(json.dumps(row), flush=True)

    try:
        with AllocatedGpuLease(worker):
            import torch
            system = next(s for s in plan["systems"] if s["name"] == "gemma3_12b")
            cls = SampleModel if a.mode == "compare" else SerialGemma
            model = cls(Path(system["path"]), system["thinking"], 20260917)
            props = torch.cuda.get_device_properties(0)
            if str(props.uuid).removeprefix("GPU-") != worker["uuid"].removeprefix("GPU-"):
                raise RuntimeError("GPU identity changed")
            write_json(a.out / "model.json", model.metadata)
            inputs = public_inputs(model, plan)
            filler = model.tokenizer.encode(" Synthetic memory stress record; no patient decision requested.", add_special_tokens=False)

            def sized(length):
                return (inputs[0] + filler * (1 + length // len(filler)))[:length]

            if a.mode == "compare":
                pairs = [inputs[i:i + 2] for i in range(0, 8, 2)] + [
                    [sized(2048), sized(1537)], [sized(4096), sized(3073)]]
                originals = []
                for i, pair in enumerate(pairs):
                    originals.append(model.generate(pair, 64))
                    progress({"kind": "baseline", "pair": i})
                install(model.model, manifest["chunk_tokens"])
                # Same loaded weights; only the separately declared execution
                # methods change. No reload or RNG reseeding for greedy tests.
                model.__class__ = SerialGemma
                fields = ("text", "input_tokens", "output_tokens", "generation_limit")
                for i, pair in enumerate(pairs):
                    fixed = model.generate(pair, 64)
                    equal = len(fixed) == len(originals[i]) and all(
                        all(old[k] == new[k] for k in fields) for old, new in zip(originals[i], fixed))
                    write_json(a.out / ("comparison" + str(i) + ".json"), {
                        "baseline": originals[i], "serial": fixed, "all_equal": equal})
                    progress({"kind": "greedy_comparison", "pair": i, "all_equal": equal})
                    if not equal:
                        raise RuntimeError("Development generation difference requires explicit review")
                for seed in (20260917, 20260918, 20260919):
                    torch.manual_seed(seed)
                    first = model.generate_sample(inputs[:2], 64, .7)
                    torch.manual_seed(seed)
                    repeat = model.generate_sample(inputs[:2], 64, .7)
                    equal = all(all(x[k] == y[k] for k in fields) for x, y in zip(first, repeat))
                    write_json(a.out / ("sampling" + str(seed) + ".json"), {
                        "seed": seed, "first": first, "repeat": repeat, "all_equal": equal,
                        "comparison_to_original_batched_sampling": False})
                    progress({"kind": "serial_sampling_reproducibility", "seed": seed, "all_equal": equal})
                    if not equal:
                        raise RuntimeError("Serial sampling is not reproducible under the recorded seed")
            else:
                # Never empty the allocator between these varied long/short
                # requests. Exercise both full output budgets twice, including
                # near-equal lengths which previously varied allocator shapes.
                probes = [(14336, 2048), (12288, 4096), (4097, 128),
                          (14335, 2048), (1024, 64), (12287, 4096)]
                for i, (length, limit) in enumerate(probes):
                    torch.cuda.reset_peak_memory_stats()
                    encoded = model.tokenizer.pad({"input_ids": [sized(length)]}, return_tensors="pt",
                        padding=True, return_attention_mask=True).to("cuda:0")
                    tick = time.monotonic()
                    with torch.inference_mode():
                        output = model.model.generate(**encoded, max_new_tokens=limit, min_new_tokens=limit,
                            do_sample=False, pad_token_id=model.tokenizer.pad_token_id, use_cache=True)
                    torch.cuda.synchronize()
                    count = output.shape[1] - encoded.input_ids.shape[1]
                    reserved = torch.cuda.max_memory_reserved()
                    record = {"kind": "sustained_serial_stress", "probe": i, "input_tokens": length,
                        "forced_new_tokens": limit, "generated_tokens": count,
                        "physical_batch_size": 1, "seconds": time.monotonic() - tick,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": reserved,
                        "gpu_total_bytes": props.total_memory, "allocator_cache_cleared_between_probes": False,
                        "all_pass": count == limit and reserved <= props.total_memory - 2 * 1024 ** 3}
                    progress(record)
                    if not record["all_pass"]:
                        raise RuntimeError("Stress probe failed full generation or 2 GiB PyTorch-reservation margin")
                    del output, encoded
            write_json(a.out / "complete.json", {"all_pass": True, "mode": a.mode,
                "manifest_hash": manifest["manifest_hash"], "worker": worker, "checks": checks,
                "held_out_inputs_used": False, "universal_equivalence_claimed": False, "time": time.time()})
    except BaseException:
        write_json(a.out / "failed.json", {"error": traceback.format_exc(), "checks": checks, "time": time.time()})
        raise


if __name__ == "__main__":
    main()
