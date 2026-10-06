"""Additive four-GPU continuation; never rewrites the completed Qwen campaign."""
import argparse
import collections
import copy
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.build import load_gzip
from medtoolbench.research.cluster_execution import (
    AllocatedGpuLease, FileLease, gpu_inventory, short_host, validate_worker)
from medtoolbench.research.job_state import write_json
from . import b_lite_execution as base
from .b_lite import ROOT, SETTINGS, read, sha
from .runner import load_plan
from .memory_execution import verify_completion as original_verify
from .b_lite_transport import (JsonModel, NativeExternalModel, NativeEpisode,
    audit_row, tokenizer_for, LLAMA_VERSION, GPT_VERSION)

CONTROL = ROOT / "data/b_lite_llama_four_gpu_v1"
ORIGINAL = ROOT / "data/b_lite_v1"
HANDOFF = ROOT / "data/b_lite_pause_after_qwen_v1"
MODELS = ("llama31_8b",)
MODULE = "medtoolbench.supplement.b_lite_four"


def clone_jobs(original):
    # Preserve the prior question ordering, within-patient shards and full cells.
    templates = [j for j in original["jobs"] if j["model"] == "qwen35_9b"]
    jobs = []
    for stage in ("qualification", "test"):
        for template in (j for j in templates if j["stage"] == stage):
            for model in MODELS:
                job = copy.deepcopy(template)
                job.update(id=job["id"].replace("qwen35_9b", model), model=model)
                jobs.append(job)
    return jobs


def prepare(control):
    control = Path(control).resolve()
    if control.exists():
        raise RuntimeError("New continuation directory must not exist")
    original = load_plan(ORIGINAL / "plan.json")
    handoff = read(HANDOFF / "ready_for_four_gpu_handoff.json")
    if not handoff or not handoff["all_pass"] or not handoff["qwen_complete"] or handoff["formal_episodes"] != 1800:
        raise RuntimeError("Qwen handoff is not complete")
    preserved = dict(read(HANDOFF / "qwen_evidence_hashes.json")["sha256"])
    changed = [p for p, checksum in preserved.items() if not Path(p).is_file() or sha(p) != checksum]
    if changed:
        raise RuntimeError("Qwen evidence changed: " + str(changed[:5]))
    # Also preserve the failed original Llama development protocol; never pool it.
    for folder in sorted((ORIGINAL / "jobs").glob("qualification--llama31_8b--*")):
        for p in folder.rglob("*.json*"):
            if p.is_file() and (p.suffix == ".json" or p.name.endswith(".json.gz")):
                preserved[str(p)] = sha(p)
    for p in (HANDOFF / "ready_for_four_gpu_handoff.json", HANDOFF / "qwen_evidence_hashes.json",
              ORIGINAL / "gates/llama31_8b.json"):
        if p.exists():
            preserved[str(p)] = sha(p)
    inventory = gpu_inventory()
    group = re.search(r"/job_(\d+)/", Path("/proc/self/cgroup").read_text())
    if len(inventory) != 4 or {g["gpu"] for g in inventory} != {0, 1, 2, 3} or not group:
        raise RuntimeError("Expected exactly four newly allocated GPUs")
    workers = [{"name": short_host() + "-gpu" + str(g["gpu"]), "host": short_host(),
                "gpu": g["gpu"], "uuid": g["uuid"], "slurm_job": int(group[1])} for g in inventory]
    for worker in workers:
        validate_worker({"workers": workers}, worker["name"])
    systems = []
    for name in MODELS:
        system = copy.deepcopy(next(s for s in original["systems"] if s["name"] == name))
        system.update(batch_size=2, inference_api=False, status="development gated four-GPU continuation")
        system["model_type"] = read(Path(system["path"]) / "config.json")["model_type"]
        system["transport"] = {"llama": LLAMA_VERSION, "gpt_oss": GPT_VERSION,
                                "mistral3": "original-json"}[system["model_type"]]
        index = read(Path(system["path"]) / "model.safetensors.index.json")
        if index and any(not (Path(system["path"]) / p).is_file() for p in set(index["weight_map"].values())):
            raise RuntimeError("Incomplete local model snapshot")
        systems.append(system)
    control.mkdir(parents=True)
    write_json(control / "legacy_inventory.json", {"files": preserved, "all_verified_at_prepare": True,
               "qwen_formal_episodes": 1800, "time": time.time()})
    code = set(original["source_hashes"])
    for name in ("b_lite_four.py", "b_lite_transport.py", "gptoss_compile.py", "gptoss_model.py",
                 "gptoss_mask.py", "gptoss_protocol.py", "gptoss_native.py"):
        code.add(str(Path(__file__).with_name(name)))
    code.add(str(ROOT / "repo/tests/test_b_lite_four.py"))
    inputs = dict(original["input_hashes"])
    input_paths = [ORIGINAL / "plan.json", control / "legacy_inventory.json",
                   HANDOFF / "ready_for_four_gpu_handoff.json", HANDOFF / "qwen_evidence_hashes.json"]
    for system in systems:
        model = Path(system["path"])
        input_paths.extend(p for p in model.iterdir() if p.is_file() and
                           (p.suffix in {".json", ".jinja", ".model"} or p.name.startswith("tekken")))
    for path in input_paths:
        inputs[str(path)] = digest(path.read_bytes().hex())
    plan = copy.deepcopy(original)
    plan.pop("plan_hash")
    plan.update(version="b-lite-four-gpu-1", created_at=time.time(), systems=systems,
        session="medtoolbench-b-lite-llama-four-v1", workers=workers, jobs=clone_jobs(original),
        source_hashes={p: digest(Path(p).read_text()) for p in sorted(code)}, input_hashes=inputs,
        parent_plan_hash=original["plan_hash"],
        amendment={"authorization": "User 2026-09-28 latest override: run only Llama first",
            "completed_qwen_rerun": False, "same_selected_questions": True,
            "original_labels_tools_parser_budgets": True, "historical_test_outcomes_exist": True,
            "sample_reselected": False, "old_rows_pooled": False,
            "protocol_strata": {s["name"]: s["transport"] for s in systems},
            "llama_original_development_failure_preserved": True,
            "llama_change": "format-only system suffix and opening-brace input prefill",
            "other_models_started": False,
            "cross_model_identical_prompt_claim": False,
            "interpretation": "Within-model paired policy effects; transport is a documented model-specific factor",
            "compute": "Llama BF16 SDPA, fixed batch 2, original local runtime",
            "inference_api": False})
    plan["plan_hash"] = digest(plan)
    write_json(control / "plan.json", plan)
    write_json(control / "matrix.json", {"plan_hash": plan["plan_hash"], "formal_episodes": 1800,
        "formal_per_model": 1800, "development_per_model": 144, "models": MODELS,
        "qwen_complete_preserved": True, "sample_source": str(ORIGINAL / "selection.json"),
        "workers": workers, "gpu_inventory": inventory})
    load(control)
    print(json.dumps({"control": str(control), "plan_hash": plan["plan_hash"], "formal_episodes": 1800}), flush=True)


def load(control):
    plan = load_plan(Path(control) / "plan.json")
    if tuple(s["name"] for s in plan["systems"]) != MODELS or any(s["batch_size"] != 2 for s in plan["systems"]):
        raise base.IntegrityError("Undeclared model or batching change")
    if any(j["settings"] != SETTINGS for j in plan["jobs"]):
        raise base.IntegrityError("Original budgets changed")
    original = read(ORIGINAL / "plan.json")
    if plan["questions"] != original["questions"] or plan["jobs"] != clone_jobs(original):
        raise base.IntegrityError("Question selection or fixed matrix changed")
    if len(plan["workers"]) != 4 or len({w["uuid"] for w in plan["workers"]}) != 4:
        raise base.IntegrityError("Invalid four-GPU allocation")
    return plan


def model_class(system):
    return NativeExternalModel if system["model_type"] == "gpt_oss" else JsonModel


def preflight(control, model_name, worker_name, output):
    plan = load(control)
    base.require_unit(control, plan)
    worker = validate_worker(plan, worker_name)
    system = next(s for s in plan["systems"] if s["name"] == model_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker["gpu"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        with AllocatedGpuLease(worker):
            write_json(output / "progress.json", {"stage": "loading", "heartbeat": time.time()})
            model = model_class(system)(Path(system["path"]), system["thinking"], SETTINGS["seed"])
            t, tok = model.torch, model.tokenizer
            if model.kind != system["model_type"] or t.cuda.device_count() != 1:
                raise base.IntegrityError("Unexpected visible model/GPU")
            device = t.cuda.get_device_properties(0)
            if str(device.uuid) != worker["uuid"]:
                raise base.IntegrityError("CUDA UUID differs from leased physical GPU")
            write_json(output / "model.json", model.metadata)
            from .external_tools import EXT_SYSTEM
            prompt = model.prompt([{"role": "system", "content": EXT_SYSTEM},
                {"role": "user", "content": "Synthetic format check, no patient data. Call metadata once."}])
            warm = model.generate([prompt, prompt], 512)
            # Full development matrix, not this prompted smoke, determines capability.
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
                "worker": worker, "cuda_uuid": str(device.uuid), "fixed_batch_size": 2,
                "prompt_tokens": 14336, "forced_output_tokens": 2048, "generated_shape": list(generated.shape),
                "seconds": seconds, "peak_reserved_bytes": reserved, "total_memory_bytes": total,
                "peak_allocated_bytes": t.cuda.max_memory_allocated(), "synthetic_only": True,
                "formal_evidence": False, "time": time.time()}
            write_json(output / "stress.json", receipt)
            if not passed:
                raise RuntimeError("Full-context memory stress/headroom failed")
            write_json(output / "complete.json", receipt)
    except BaseException:
        write_json(output / "failure.json", {"error": traceback.format_exc(), "time": time.time()})
        raise


def transport_audit(output, plan, job):
    system = next(s for s in plan["systems"] if s["name"] == job["model"])
    tokenizer, template = tokenizer_for(system)
    seen, errors, files, statuses = set(), [], {}, collections.Counter()
    for path in sorted((output / "episodes").glob("*.json.gz")):
        row = load_gzip(path)
        if row["key"] in seen:
            errors.append({"key": row["key"], "error": "duplicate"})
        seen.add(row["key"])
        files[path.name] = sha(path)
        try:
            statuses.update(audit_row(row, tokenizer, system["model_type"], template))
        except Exception:
            errors.append({"key": row["key"], "error": traceback.format_exc()})
    expected = {e["key"] for e in job["episodes"]}
    proof = {"all_pass": seen == expected and not errors, "plan_hash": plan["plan_hash"],
        "job": job["id"], "episodes": len(seen), "files": files, "errors": errors,
        "status_counts": dict(statuses), "missing": sorted(expected - seen),
        "unexpected": sorted(seen - expected), "exact_prompt_token_replay": True,
        "raw_output_replay": True, "protocol": system["transport"]}
    write_json(output / "transport_audit.json", proof)
    if not proof["all_pass"]:
        raise base.IntegrityError("Independent transport replay failed")
    return proof


def verify_completion(output, plan, job):
    original_verify(output, plan, job)
    proof = read(output / "transport_audit.json")
    if not proof or not proof.get("all_pass") or proof["plan_hash"] != plan["plan_hash"] or proof["job"] != job["id"]:
        raise base.IntegrityError("Missing or mismatched raw transport audit")
    expected = {e["key"] + ".json.gz" for e in job["episodes"]}
    if set(proof["files"]) != expected or proof["episodes"] != len(expected):
        raise base.IntegrityError("Transport coverage differs from assignment")
    actual = {p.name for p in (output / "episodes").glob("*.json.gz")}
    if actual != expected or any(sha(output / "episodes" / p) != h for p, h in proof["files"].items()):
        raise base.IntegrityError("Audited trajectory was changed")


def run_shard(control, job_id, worker_name, output):
    plan = load(control)
    base.require_unit(control, plan)
    job = next(j for j in plan["jobs"] if j["id"] == job_id)
    if job["stage"] == "test" and base.model_gate(control, plan, job["model"]) != "passed":
        raise base.IntegrityError("Held-out execution cannot bypass development gates")
    system = next(s for s in plan["systems"] if s["name"] == job["model"])
    from . import external_runner
    external_runner.SampleModel = model_class(system)
    if system["model_type"] == "gpt_oss":
        external_runner.Episode = NativeEpisode
    sys.argv = [external_runner.__file__, "--plan", str(Path(control) / "plan.json"),
                "--job", job_id, "--worker", worker_name, "--out", str(output)]
    external_runner.main()
    # A legacy complete.json alone is NOT sufficient for this continuation.
    transport_audit(output, plan, job)


def attempt(control, job, worker, owner_fd, stop):
    control = Path(control)
    plan = load(control)
    pre = job["stage"] == "preflight"
    folder = control / "preflight" / job["model"] if pre else control / "jobs" / job["id"]
    with FileLease(control / "locks" / (job["id"] + ".lock")) as lease:
        if folder.exists():
            raise base.IntegrityError("Existing unresolved attempt: " + str(folder))
        folder.mkdir(parents=True)
        output = folder / "run" if pre else folder / "attempt1"
        command = [plan["python"], "-u", "-m", MODULE, "preflight" if pre else "shard",
            "--control", str(control), "--worker", worker["name"], "--out", str(output)]
        command += ["--model", job["model"]] if pre else ["--job", job["id"]]
        write_json(folder / "command.json", {"command": command, "plan_hash": plan["plan_hash"],
            "worker": worker, "time": time.time(), "allocator": "expandable_segments:True"})
        execution = base.execute(command, folder / "run.log", control / "workers" / (worker["name"] + ".json"),
            worker, [owner_fd, lease.fileno()], stop,
            (plan["execution"]["stall_seconds"], plan["execution"]["wall_seconds"]))
        result = {"job": job["id"], "model": job["model"], "stage": job["stage"], "run": str(output),
            "worker": worker, "execution": execution, "plan_hash": plan["plan_hash"], "time": time.time()}
        completion = read(output / "complete.json")
        if execution["returncode"] == 0 and completion and completion.get("all_pass"):
            if pre:
                if completion["plan_hash"] != plan["plan_hash"]:
                    raise base.IntegrityError("Stress receipt identity mismatch")
                write_json(folder / "complete.json", completion)
            else:
                from .external_runner import audit
                if not audit(output, plan, job)["all_pass"]:
                    raise base.IntegrityError("Original independent re-audit failed")
                verify_completion(output, plan, job)
            result.update(state="complete", episodes=0 if pre else len(job["episodes"]))
        else:
            if not pre and ((output / "audit.json").exists() or completion):
                raise base.IntegrityError("Completed/audit-invalid evidence needs global review")
            failure = read(output / "failure.json") or {"error": str(execution)}
            result.update(state="quarantined", assigned_episodes=0 if pre else len(job["episodes"]),
                failure=failure, partial_rows_counted=False,
                failure_kind="cuda_oom" if "CUDA out of memory" in failure["error"] else "runtime_failure")
            if pre:
                write_json(folder / "failure.json", failure)
        write_json(folder / "result.json", result)
        write_json(control / "workers" / (worker["name"] + ".json"),
            {"state": "idle", "job": job["id"], "result": result["state"], "heartbeat": time.time()})
        return result


def install():
    # Process-local wiring: the original frozen files and campaign stay untouched.
    base.MODELS, base.CONTROL = MODELS, CONTROL
    base.load, base.attempt, base.verify_completion = load, attempt, verify_completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "run", "preflight", "shard", "report"))
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--worker")
    parser.add_argument("--job")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    install()
    if args.action == "prepare":
        prepare(args.control)
    elif args.action == "validate":
        base.validate(args.control)
    elif args.action == "preflight":
        preflight(args.control, args.model, args.worker, args.out)
    elif args.action == "shard":
        run_shard(args.control, args.job, args.worker, args.out)
    elif args.action == "report":
        base.report(args.control, load(args.control))
    else:
        try:
            base.run(args.control, args.resume)
        except BaseException:
            write_json(args.control / "controller_failure.json", {"error": traceback.format_exc(), "time": time.time()})
            raise


if __name__ == "__main__":
    main()
