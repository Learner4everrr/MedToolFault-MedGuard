"""Frozen B-lite extension: Ministral then GPT-OSS, without interrupting Llama.

Seven lanes are available immediately; lane 7 joins only after the independent
Llama controller certifies completion and releases its ownership lock. Original
tools, labels, question matrix, budgets, audit and within-model analysis remain
unchanged. New models must each pass memory and complete development gates.
"""
import argparse
import collections
import copy
import json
import os
import signal
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import FileLease, AllocatedGpuLease, validate_worker
from medtoolbench.research.job_state import write_json
from . import b_lite_four as four
from . import b_lite_four_identity as identity
from .b_lite import ROOT, SETTINGS, read, sha
from .b_lite_transport import JsonModel, NativeExternalModel, GPT_VERSION
from .runner import load_plan

CONTROL = ROOT / "data/b_lite_remaining_models_v1"
LLAMA = ROOT / "data/b_lite_llama_eight_gpu_v1"
ORIGINAL = ROOT / "data/b_lite_v1"
MODELS = ("ministral3_8b", "gptoss20b")
MODULE = "medtoolbench.supplement.b_lite_remaining"
BASE_ELIGIBLE = four.base.eligible
BASE_ATTEMPT = four.attempt
BASE_SHARD = four.run_shard
MODEL_SOURCES = (ROOT / "data/supplement_b_v2/plan.json",
                 ROOT / "data/gptoss_compile_recovery_v1/campaign_a/plan.json")


def clone_jobs(original):
    jobs = []
    for model in MODELS:
        for template in original["jobs"]:
            if template["model"] == "qwen35_9b":
                job = copy.deepcopy(template)
                job.update(id=job["id"].replace("qwen35_9b", model), model=model)
                jobs.append(job)
    return jobs


def assert_assignment(plan, original):
    if tuple(s["name"] for s in plan["systems"]) != MODELS:
        raise four.base.IntegrityError("Undeclared remaining-model substitution")
    if any(s["batch_size"] != 2 or s.get("inference_api") is not False for s in plan["systems"]):
        raise four.base.IntegrityError("Batch or local-inference contract changed")
    if plan["questions"] != original["questions"] or plan["jobs"] != clone_jobs(original):
        raise four.base.IntegrityError("Original question assignment changed")
    for key in ("statistics", "execution", "sources", "scope", "adaptations", "image", "python"):
        if plan[key] != original[key]:
            raise four.base.IntegrityError("Scientific contract changed: " + key)
    if any(j["settings"] != SETTINGS for j in plan["jobs"]):
        raise four.base.IntegrityError("Original per-episode budgets changed")
    workers = plan["workers"]
    if (len(workers) != 8 or {w["gpu"] for w in workers} != set(range(8)) or
            len({w["uuid"] for w in workers}) != 8 or len({w["name"] for w in workers}) != 8):
        raise four.base.IntegrityError("Eight distinct physical GPU lanes required")


def load(control):
    plan = load_plan(Path(control) / "plan.json")
    assert_assignment(plan, read(ORIGINAL / "plan.json"))
    return plan


def prepare(control):
    control = Path(control).resolve()
    if control.exists():
        raise RuntimeError("Never overwrite an experiment campaign")
    original = load_plan(ORIGINAL / "plan.json")
    llama = load_plan(LLAMA / "plan.json")
    workers = copy.deepcopy(llama["workers"])
    for worker in workers:
        validate_worker({"workers": workers}, worker["name"])
    handoff = ROOT / "data/b_lite_pause_after_qwen_v1"
    preserved = dict(read(handoff / "qwen_evidence_hashes.json")["sha256"])
    preserved[str(LLAMA / "plan.json")] = sha(LLAMA / "plan.json")
    if any(not Path(p).is_file() or sha(p) != h for p, h in preserved.items()):
        raise RuntimeError("Historical evidence changed")
    systems = []
    for name, source in zip(MODELS, MODEL_SOURCES):
        system = copy.deepcopy(next(s for s in read(source)["systems"] if s["name"] == name))
        model = Path(system["path"])
        system.update(batch_size=2, inference_api=False, model_type=read(model / "config.json")["model_type"],
                      status="own memory and complete development gates required")
        system["transport"] = GPT_VERSION if system["model_type"] == "gpt_oss" else "original-json"
        index = read(model / "model.safetensors.index.json")
        if not index or any(not (model / p).is_file() for p in set(index["weight_map"].values())):
            raise RuntimeError("Local model snapshot is incomplete")
        systems.append(system)
    control.mkdir(parents=True)
    write_json(control / "legacy_inventory.json", {"files": preserved, "all_verified_at_prepare": True,
        "ongoing_llama_outputs_not_frozen_or_modified": True, "time": time.time()})
    code = set(original["source_hashes"]) | set(llama["source_hashes"])
    code.update(str(p) for p in (Path(__file__), ROOT / "repo/tests/test_b_lite_remaining.py"))
    inputs = dict(original["input_hashes"])
    paths = [ORIGINAL / "plan.json", LLAMA / "plan.json", control / "legacy_inventory.json", *MODEL_SOURCES]
    for system in systems:
        paths.extend(p for p in Path(system["path"]).iterdir() if p.is_file() and
                     (p.suffix in {".json", ".jinja", ".model"} or p.name.startswith("tekken")))
    for p in paths:
        inputs[str(p)] = digest(p.read_bytes().hex())
    plan = copy.deepcopy(original)
    plan.pop("plan_hash")
    plan.update(version="b-lite-remaining-models-1", created_at=time.time(), systems=systems,
        workers=workers, jobs=clone_jobs(original), session="medtoolbench-b-lite-remaining-v1",
        source_hashes={p: digest(Path(p).read_text()) for p in sorted(code)}, input_hashes=inputs,
        parent_plan_hash=original["plan_hash"],
        scheduling={"model_order": list(MODELS), "initial_gpus": list(range(7)),
            "late_gpu": 7, "dependency_control": str(LLAMA), "dependency_plan_hash": llama["plan_hash"],
            "join_requires": "complete receipt, complete report and released supervisor; exclusive physical lease",
            "blocked_model_policy": "finish its active shards, then continue independent next model; never accuracy-retry",
            "durable_drain": "PAUSE_REQUESTED.json blocks selection and launch; in-flight shards finish"},
        amendment={"authorization": "User 2026-09-29: remaining two models may start now",
            "same_selected_questions": True, "original_labels_tools_parser_budgets": True,
            "sample_reselected": False, "old_rows_pooled": False, "qwen_or_llama_rerun": False,
            "historical_test_outcomes_exist": True, "inference_api": False,
            "protocol_strata": {s["name"]: s["transport"] for s in systems},
            "cross_model_identical_prompt_claim": False,
            "interpretation": "Within-model paired effects; model-specific serialization is a documented factor",
            "gptoss_compute": "native MXFP4, low reasoning, pinned local kernels and validated compile/mask repairs",
            "ministral_compute": "existing BF16 dequantization runtime, original JSON delivery",
            "qualification": "each model: synthetic batch-2 16384-token memory stress, then 144 audited development episodes",
            "clinical_accuracy_gate": False})
    assert_assignment(plan, original)
    plan["plan_hash"] = digest(plan)
    write_json(control / "plan.json", plan)
    write_json(control / "matrix.json", {"plan_hash": plan["plan_hash"], "models": MODELS,
        "formal_per_model": 1800, "development_per_model": 144, "formal_total": 3600,
        "workers": workers, "sample_source": str(ORIGINAL / "selection.json")})
    load(control)
    print(json.dumps({"control": str(control), "plan_hash": plan["plan_hash"], "formal_total": 3600}), flush=True)


def drain_requested(control, plan):
    marker = read(Path(control) / "PAUSE_REQUESTED.json")
    if marker is None:
        return False
    if marker.get("action") != "drain" or marker.get("plan_hash") != plan["plan_hash"]:
        raise four.base.IntegrityError("Invalid durable drain marker")
    return True


def selected_model(control, plan, active, blocked):
    # Do not overlap model families, including when the prior family is blocked
    # but still draining its in-flight shards.
    for model in MODELS:
        if any(key.endswith("--" + model) or ("--" + model + "--") in key for key in active):
            return model
        if model not in blocked and any(not (Path(control) / "jobs" / j["id"] / "result.json").exists()
                                       for j in plan["jobs"] if j["model"] == model):
            return model
    return None


def eligible(control, plan, active, blocked):
    if drain_requested(control, plan):
        return []
    model = selected_model(control, plan, active, blocked)
    return [j for j in BASE_ELIGIBLE(control, plan, active, blocked) if j["model"] == model]


def late_gpu_ready(plan):
    dep = Path(plan["scheduling"]["dependency_control"])
    receipt = read(dep / "complete.json")
    if receipt is None:
        return False
    expected = plan["scheduling"]["dependency_plan_hash"]
    report = read(dep / "report/metrics.json")
    if (receipt.get("plan_hash") != expected or not receipt.get("all_pass") or
            receipt.get("formal_episodes") != 1800 or not report or not report.get("complete") or
            report.get("plan_hash") != expected):
        raise four.base.IntegrityError("Llama completion dependency identity/coverage invalid")
    try:
        with FileLease(dep / "supervisor.lock"):
            return True
    except BlockingIOError:
        return False


def available_workers(plan):
    ready = late_gpu_ready(plan)
    return [w for w in plan["workers"] if w["gpu"] != plan["scheduling"]["late_gpu"] or ready]


def bound_model_class(system, worker, plan):
    parent = NativeExternalModel if system["model_type"] == "gpt_oss" else JsonModel

    class BoundModel(parent):
        def __init__(self, *args, **kwargs):
            import torch
            if torch.cuda.device_count() != 1 or not identity.same_device(torch.cuda.get_device_properties(0).uuid, worker["uuid"]):
                raise four.base.IntegrityError("CUDA device differs from assigned physical GPU")
            super().__init__(*args, **kwargs)
            if self.kind != system["model_type"]:
                raise four.base.IntegrityError("Loaded architecture differs from frozen assignment")
            self.metadata.update(b_lite_plan_hash=plan["plan_hash"], allocated_worker=worker,
                clinical_tools_parser_labels_budgets_unchanged=True, fixed_batch_size=2)
            if self.kind == "gpt_oss":
                self.metadata["protocol"] = "declared native Harmony external tools; original JSON final; all generated tokens budgeted"
    return BoundModel


def bind(control, worker_name):
    plan = load(control)
    worker = validate_worker(plan, worker_name)
    if worker["gpu"] == plan["scheduling"]["late_gpu"] and not late_gpu_ready(plan):
        raise four.base.IntegrityError("Llama still owns the deferred GPU")
    four.model_class = lambda system: bound_model_class(system, worker, plan)


def preflight(control, model, worker, output):
    bind(control, worker)
    identity.preflight(control, model, worker, output)


def run_shard(control, job, worker, output):
    bind(control, worker)
    BASE_SHARD(control, job, worker, output)


def attempt(control, job, worker, owner_fd, stop):
    plan = load(control)
    if drain_requested(control, plan):
        return {"state": "not_dispatched", "job": job["id"], "model": job["model"], "stage": job["stage"]}
    if worker["gpu"] == plan["scheduling"]["late_gpu"] and not late_gpu_ready(plan):
        raise four.base.IntegrityError("Deferred GPU dependency not satisfied")
    return BASE_ATTEMPT(control, job, worker, owner_fd, stop)


def run(control, resume=False):
    control = Path(control).resolve()
    plan = load(control)
    base = four.base
    base.require_unit(control, plan)
    for worker in plan["workers"]:
        validate_worker(plan, worker["name"])
    for name in ("workers", "preflight", "jobs", "locks", "sessions"):
        (control / name).mkdir(exist_ok=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with FileLease(control / "supervisor.lock") as owner:
        if (control / "complete.json").exists():
            raise base.IntegrityError("Campaign already complete")
        previous = read(control / "started.json")
        if previous and not resume:
            raise base.IntegrityError("Explicit checkpoint reconciliation required")
        old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            for worker in available_workers(plan):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(worker["gpu"])
                with AllocatedGpuLease(worker):
                    pass
        finally:
            if old_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
        done, failed, pending = base.inventory(control, plan)
        for job, result in done:
            four.verify_completion(Path(result["run"]), plan, job)
        for job in pending:
            if (control / "jobs" / job["id"]).exists():
                raise base.IntegrityError("Unresolved checkpoint: " + job["id"])
        for model in MODELS:
            folder = control / "preflight" / model
            if folder.exists() and not (folder / "result.json").exists():
                raise base.IntegrityError("Unresolved preflight: " + model)
        baseline = sum(len(j["episodes"]) for j, _ in done if j["stage"] == "test")
        started = time.time()
        session = {"pid": os.getpid(), "time": started, "plan_hash": plan["plan_hash"],
            "resume": resume, "prior_started": previous, "completed_formal_checkpoint": baseline}
        write_json(control / "sessions" / (str(time.time_ns()) + ".json"), session)
        if not previous:
            write_json(control / "started.json", session)
        blocked, busy, fatal = {}, {}, None
        for model in MODELS:
            if any(j["model"] == model and j["stage"] == "qualification" for j, _ in failed):
                blocked[model] = "qualification failure"
            if (control / "preflight" / model / "failure.json").exists():
                blocked[model] = "preflight failure"
        with ThreadPoolExecutor(max_workers=8) as pool:
            while True:
                try:
                    load(control)
                    for future, (job, worker) in list(busy.items()):
                        if future.done():
                            del busy[future]
                            result = future.result()
                            if result["state"] not in {"complete", "not_dispatched"} and job["stage"] in {"preflight", "qualification"}:
                                blocked[job["model"]] = job["stage"] + " failed"
                    _, failures, _ = base.inventory(control, plan)
                    for model, count in collections.Counter(j["model"] for j, _ in failures).items():
                        if count >= plan["execution"]["model_circuit_breaker_failures"]:
                            blocked[model] = "three failed shards; model circuit breaker"
                    for model in MODELS:
                        if model not in blocked and base.model_gate(control, plan, model) == "failed":
                            blocked[model] = "development gate failed"
                    if not fatal and not stop.is_set():
                        choices = eligible(control, plan, {j["id"] for j, _ in busy.values()}, blocked)
                        occupied = {w["name"] for _, w in busy.values()}
                        for worker in available_workers(plan):
                            if worker["name"] not in occupied and choices:
                                job = choices.pop(0)
                                busy[pool.submit(attempt, control, job, worker, owner.fileno(), stop)] = (job, worker)
                except Exception:
                    fatal = traceback.format_exc()
                    write_json(control / "integrity_failure.json", {"error": fatal, "time": time.time()})
                state = base.snapshot(control, plan, busy, blocked, started, baseline, fatal)
                state.update(model_order=list(MODELS), durable_drain_requested=drain_requested(control, plan),
                    deferred_gpu=plan["scheduling"]["late_gpu"],
                    eta_note="Do not extrapolate one model's throughput to the next model.")
                write_json(control / "status.json", state)
                if not busy and (fatal or stop.is_set() or not eligible(control, plan, set(), blocked)):
                    break
                time.sleep(10)
        full = False if fatal or stop.is_set() else base.report(control, plan)
        if full:
            changed = [p for p, h in read(control / "legacy_inventory.json")["files"].items()
                       if not Path(p).is_file() or sha(p) != h]
            if changed:
                raise base.IntegrityError("Historical evidence changed: " + str(changed[:5]))
            write_json(control / "complete.json", {"all_pass": True, "plan_hash": plan["plan_hash"],
                "formal_episodes": state["formal_total"], "legacy_evidence_unchanged": True,
                "entire_project_complete": False, "time": time.time()})
        state.update(state="complete" if full else "drained" if drain_requested(control, plan) else "needs_review",
                     active=[], heartbeat=time.time())
        write_json(control / "status.json", state)


def install():
    four.CONTROL, four.MODULE, four.load = CONTROL, MODULE, load
    four.base.MODELS, four.base.CONTROL = MODELS, CONTROL
    four.base.load, four.base.attempt = load, attempt
    four.base.verify_completion, four.base.eligible = four.verify_completion, eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "run", "preflight", "shard", "report", "drain"))
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--worker")
    parser.add_argument("--job")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.control)
        return
    install()
    if args.action == "validate":
        four.base.validate(args.control)
    elif args.action == "preflight":
        preflight(args.control, args.model, args.worker, args.out)
    elif args.action == "shard":
        run_shard(args.control, args.job, args.worker, args.out)
    elif args.action == "report":
        four.base.report(args.control, load(args.control))
    elif args.action == "drain":
        plan = load(args.control)
        if (args.control / "PAUSE_REQUESTED.json").exists():
            raise RuntimeError("Drain already requested")
        write_json(args.control / "PAUSE_REQUESTED.json", {"action": "drain", "plan_hash": plan["plan_hash"],
            "time": time.time(), "instruction": "Finish in-flight shards, dispatch nothing else; no automatic resume"})
    else:
        try:
            run(args.control, args.resume)
        except BaseException:
            write_json(args.control / "controller_failure.json", {"error": traceback.format_exc(), "time": time.time()})
            raise


if __name__ == "__main__":
    main()
