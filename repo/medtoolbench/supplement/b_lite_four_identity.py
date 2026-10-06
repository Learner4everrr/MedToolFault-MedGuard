"""Resource-only correction: PyTorch UUIDs omit nvidia-smi's GPU- prefix.

The initial failed preflight, plan and source remain immutable. No episode ran.
"""
import copy
import json
import os
import time
import traceback
import uuid
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import AllocatedGpuLease, validate_worker
from medtoolbench.research.job_state import write_json
from . import b_lite_four as four
from .b_lite import ROOT, SETTINGS, read, sha

PREVIOUS = ROOT / "data/b_lite_llama_four_gpu_v1"
CONTROL = ROOT / "data/b_lite_llama_four_gpu_v2"
MODULE = "medtoolbench.supplement.b_lite_four_identity"


def canonical_uuid(value):
    return str(uuid.UUID(str(value).removeprefix("GPU-")))


def same_device(actual, allocated):
    return canonical_uuid(actual) == canonical_uuid(allocated)


def prepare(control):
    control = Path(control).resolve()
    if control.exists():
        raise RuntimeError("Never overwrite a continuation")
    original = four.load(PREVIOUS)
    status = read(PREVIOUS / "status.json")
    failure = read(PREVIOUS / "preflight/llama31_8b/run/failure.json")
    if not status or status["active"] or status["audited_formal"] or status["audited_development"]:
        raise RuntimeError("Only the terminal pre-episode UUID failure is eligible")
    if not failure or "CUDA UUID differs from leased physical GPU" not in failure["error"]:
        raise RuntimeError("Unexpected failure; not a resource-format correction")
    if any((PREVIOUS / "jobs").iterdir()):
        raise RuntimeError("Cannot use this pre-episode amendment after development dispatch")
    for worker in original["workers"]:
        validate_worker(original, worker["name"])
    preserved = dict(read(PREVIOUS / "legacy_inventory.json")["files"])
    for p in PREVIOUS.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".log"}:
            preserved[str(p)] = sha(p)
    if any(not Path(p).is_file() or sha(p) != value for p, value in preserved.items()):
        raise RuntimeError("Historical evidence changed")
    control.mkdir(parents=True)
    write_json(control / "legacy_inventory.json", {"files": preserved, "all_verified_at_prepare": True,
        "qwen_formal_episodes": 1800, "prior_preflight_preserved": True, "time": time.time()})
    plan = copy.deepcopy(original)
    plan.pop("plan_hash")
    plan.update(version="b-lite-llama-four-gpu-2", created_at=time.time(),
                session="medtoolbench-b-lite-llama-four-v2")
    plan["resource_amendment"] = {"previous_plan_hash": original["plan_hash"],
        "change": "Compare canonical UUID values, accepting the GPU- display prefix only",
        "architecture_precision_protocol_batch_budgets_questions_unchanged": True,
        "only_model": "llama31_8b", "entry_point": MODULE}
    for path in (Path(__file__), ROOT / "repo/tests/test_b_lite_four_identity.py"):
        plan["source_hashes"][str(path)] = digest(path.read_text())
    for path in (PREVIOUS / "plan.json", control / "legacy_inventory.json"):
        plan["input_hashes"][str(path)] = digest(path.read_bytes().hex())
    plan["plan_hash"] = digest(plan)
    write_json(control / "plan.json", plan)
    four.load(control)
    print(json.dumps({"control": str(control), "plan_hash": plan["plan_hash"], "formal_episodes": 1800}), flush=True)


def preflight(control, model_name, worker_name, output):
    # Same preflight as v1, with only canonical UUID comparison corrected.
    plan = four.load(control)
    four.base.require_unit(control, plan)
    worker = validate_worker(plan, worker_name)
    system = next(s for s in plan["systems"] if s["name"] == model_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker["gpu"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        with AllocatedGpuLease(worker):
            write_json(output / "progress.json", {"stage": "loading", "heartbeat": time.time()})
            model = four.model_class(system)(Path(system["path"]), system["thinking"], SETTINGS["seed"])
            t, tok = model.torch, model.tokenizer
            if model.kind != system["model_type"] or t.cuda.device_count() != 1:
                raise four.base.IntegrityError("Unexpected visible model/GPU")
            device = t.cuda.get_device_properties(0)
            if not same_device(device.uuid, worker["uuid"]):
                raise four.base.IntegrityError("CUDA physical UUID differs from leased GPU")
            write_json(output / "model.json", model.metadata)
            from .external_tools import EXT_SYSTEM
            prompt = model.prompt([{"role": "system", "content": EXT_SYSTEM},
                {"role": "user", "content": "Synthetic format check, no patient data. Call metadata once."}])
            warm = model.generate([prompt, prompt], 512)
            write_json(output / "warmup.json", {"generations": warm, "synthetic_only": True})
            write_json(output / "progress.json", {"stage": "memory_stress", "heartbeat": time.time()})
            token = tok.encode(" record", add_special_tokens=False)[0]
            ids = t.full((2, 14336), token, dtype=t.long, device="cuda:0")
            mask = t.ones_like(ids)
            t.cuda.synchronize()
            t.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with t.inference_mode():
                generated = model.model.generate(input_ids=ids, attention_mask=mask,
                    max_new_tokens=2048, min_new_tokens=2048, do_sample=False,
                    eos_token_id=None, pad_token_id=tok.pad_token_id, use_cache=True)
            t.cuda.synchronize()
            seconds = time.perf_counter() - started
            reserved, total = t.cuda.max_memory_reserved(), device.total_memory
            passed = tuple(generated.shape) == (2, 16384) and reserved / total <= .95
            receipt = {"all_pass": passed, "plan_hash": plan["plan_hash"], "model": model_name,
                "worker": worker, "cuda_uuid": str(device.uuid), "canonical_uuid": canonical_uuid(device.uuid),
                "fixed_batch_size": 2, "prompt_tokens": 14336, "forced_output_tokens": 2048,
                "generated_shape": list(generated.shape), "seconds": seconds,
                "peak_reserved_bytes": reserved, "total_memory_bytes": total,
                "peak_allocated_bytes": t.cuda.max_memory_allocated(), "synthetic_only": True,
                "formal_evidence": False, "time": time.time()}
            write_json(output / "stress.json", receipt)
            if not passed:
                raise RuntimeError("Full-context memory stress/headroom failed")
            write_json(output / "complete.json", receipt)
    except BaseException:
        write_json(output / "failure.json", {"error": traceback.format_exc(), "time": time.time()})
        raise


def main():
    four.CONTROL, four.MODULE = CONTROL, MODULE
    four.prepare, four.preflight = prepare, preflight
    four.main()


if __name__ == "__main__":
    main()
