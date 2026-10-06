"""Eight-GPU, Llama-only continuation with immutable checkpoint adoption.

Same scientific plan/adapter/batch/budgets. Completed shards are referenced,
never relabeled or rerun. Administrative interruptions restart as whole shards;
their partial rows remain excluded. A durable drain marker prevents new work.
"""
import argparse
import copy
import json
import os
import re
import time
import traceback
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import (
    FileLease, AllocatedGpuLease, gpu_inventory, short_host, validate_worker)
from medtoolbench.research.job_state import write_json
from . import b_lite_four as four
from .b_lite import ROOT, read, sha
from .b_lite_four_identity import same_device
from .runner import load_plan

CONTROL = ROOT / "data/b_lite_llama_eight_gpu_v1"
PREVIOUS = ROOT / "data/b_lite_llama_four_gpu_v2"
HANDOFF = ROOT / "data/b_lite_pause_after_llama_dev_v1"
MODULE = "medtoolbench.supplement.b_lite_eight"
OLD_LOAD = four.load
OLD_VERIFY = four.verify_completion
OLD_ATTEMPT = four.attempt
OLD_ELIGIBLE = four.base.eligible
OLD_RUN_SHARD = four.run_shard
SCIENCE_KEYS = ("python", "image", "primary_core_hashes", "sources", "scope", "adaptations",
                "inference_api", "runner_module", "systems", "questions", "jobs", "statistics",
                "execution", "amendment", "parent_plan_hash")


def assert_science(plan, previous):
    for key in SCIENCE_KEYS:
        if plan[key] != previous[key]:
            raise RuntimeError("Resource resume changed scientific field: " + key)
    if any(plan["source_hashes"].get(p) != h for p, h in previous["source_hashes"].items()):
        raise RuntimeError("Frozen source was modified during resource resume")
    if [s["name"] for s in plan["systems"]] != ["llama31_8b"]:
        raise RuntimeError("Only Llama is authorized")


def administrative_interruption(result, output, job):
    if (job["stage"] != "test" or result.get("state") != "quarantined" or
            result.get("execution", {}).get("timeout") != "controller_stop"):
        raise RuntimeError("Only documented administrative interruption qualifies for whole-shard restart")
    if result.get("plan_hash") is None or result.get("job") != job["id"] or result.get("partial_rows_counted") is not False:
        raise RuntimeError("Invalid interrupted result identity")
    if any((output / n).exists() for n in ("complete.json", "audit.json", "transport_audit.json")):
        raise RuntimeError("Never rerun completed/audit-invalid evidence based on outcomes")


def load(control):
    plan = load_plan(Path(control) / "plan.json")
    prior = read(PREVIOUS / "plan.json")
    assert_science(plan, prior)
    workers = plan["workers"]
    if (len(workers) != 8 or {w["gpu"] for w in workers} != set(range(8)) or
            len({w["uuid"] for w in workers}) != 8 or len({w["name"] for w in workers}) != 8):
        raise RuntimeError("Expected eight distinct physical GPU lanes")
    if plan["resume"]["previous_plan_hash"] != prior["plan_hash"]:
        raise RuntimeError("Checkpoint plan identity mismatch")
    for job_id, reference in plan["resume"]["adopted_shards"].items():
        result = read(Path(control) / "jobs" / job_id / "result.json")
        if (not result or result.get("state") != "complete" or result.get("plan_hash") != plan["plan_hash"] or
                result.get("run") != reference["run"] or result.get("episodes") != reference["episodes"] or
                result.get("adopted_from_plan_hash") != prior["plan_hash"]):
            raise RuntimeError("Adopted completion receipt changed")
    stress = read(Path(control) / "preflight/llama31_8b/complete.json")
    if (not stress or not stress.get("all_pass") or stress.get("plan_hash") != plan["plan_hash"] or
            stress.get("original_receipt") != plan["resume"]["stress_inherited_from"] or
            stress.get("original_sha256") != plan["resume"]["stress_inherited_sha256"] or
            stress.get("new_stress_run") is not False):
        raise RuntimeError("Inherited memory gate identity changed")
    return plan


def prepare(control):
    control = Path(control).resolve()
    if control.exists():
        raise RuntimeError("Never overwrite a resume campaign")
    previous = OLD_LOAD(PREVIOUS)
    handoff = read(HANDOFF / "ready_for_eight_gpu_handoff.json")
    if (not handoff or not handoff["controller_exited"] or not handoff["job_and_gpu_leases_released"] or
            handoff["development_audited"] != 144 or not handoff["development_gate_passed"] or
            handoff["plan_hash"] != previous["plan_hash"]):
        raise RuntimeError("Invalid eight-GPU handoff")
    preserved = dict(read(PREVIOUS / "legacy_inventory.json")["files"])
    preserved.update(read(HANDOFF / "evidence_hashes.json")["sha256"])
    for path in (HANDOFF / "ready_for_eight_gpu_handoff.json", HANDOFF / "evidence_hashes.json",
                 PREVIOUS / "preflight/llama31_8b/complete.json"):
        preserved[str(path)] = sha(path)
    changed = [p for p, value in preserved.items() if not Path(p).is_file() or sha(p) != value]
    if changed:
        raise RuntimeError("Certified evidence changed: " + str(changed[:5]))
    inventory = gpu_inventory()
    group = re.search(r"/job_(\d+)/", Path("/proc/self/cgroup").read_text())
    if len(inventory) != 8 or not group or any(g["name"] != "NVIDIA A100-SXM4-40GB" for g in inventory):
        raise RuntimeError("Resume requires the declared eight matching A100 40GB GPUs")
    workers = [{"name": short_host() + "-gpu" + str(g["gpu"]), "host": short_host(), "gpu": g["gpu"],
                "uuid": g["uuid"], "slurm_job": int(group[1])} for g in inventory]
    for worker in workers:
        validate_worker({"workers": workers}, worker["name"])
    stress_path = PREVIOUS / "preflight/llama31_8b/complete.json"
    stress = read(stress_path)
    if (not stress["all_pass"] or stress["plan_hash"] != previous["plan_hash"] or
            stress["fixed_batch_size"] != 2 or stress["generated_shape"] != [2, 16384]):
        raise RuntimeError("Original identical-hardware full-context stress gate is not valid")
    adopted, interruptions = {}, {}
    with FileLease(PREVIOUS / "supervisor.lock"):
        for job in previous["jobs"]:
            folder = PREVIOUS / "jobs" / job["id"]
            result = read(folder / "result.json")
            if not result:
                if folder.exists() or job["stage"] == "qualification":
                    raise RuntimeError("Unresolved old checkpoint")
                continue
            with FileLease(PREVIOUS / "locks" / (job["id"] + ".lock")):
                output = Path(result["run"])
                if result["state"] == "complete":
                    OLD_VERIFY(output, previous, job)
                    adopted[job["id"]] = {"run": str(output), "result": str(folder / "result.json"),
                        "previous_plan_hash": previous["plan_hash"], "episodes": len(job["episodes"]),
                        "stage": job["stage"], "files": {p: h for p, h in preserved.items()
                            if Path(p).is_relative_to(folder)}}
                else:
                    administrative_interruption(result, output, job)
                    if result["plan_hash"] != previous["plan_hash"]:
                        raise RuntimeError("Interrupted evidence belongs to another plan")
                    interruptions[job["id"]] = {"original_result": str(folder / "result.json"),
                        "original_result_sha256": sha(folder / "result.json"),
                        "assigned_episodes": len(job["episodes"]),
                        "saved_partial": len(list((output / "episodes").glob("*.json.gz"))),
                        "policy": "restart entire original shard once; all old partial rows excluded"}
    if (sum(r["episodes"] for r in adopted.values() if r["stage"] == "qualification") != 144 or
            sum(r["episodes"] for r in adopted.values() if r["stage"] == "test") != 18 or
            set(interruptions) != {r["job"] for r in handoff["interrupted_shards"]}):
        raise RuntimeError("Checkpoint counts differ from the certified handoff")
    control.mkdir(parents=True)
    write_json(control / "legacy_inventory.json", {"files": preserved, "all_verified_at_prepare": True,
        "time": time.time(), "partial_rows_pooled": False})
    plan = copy.deepcopy(previous)
    plan.pop("plan_hash")
    plan.update(version="b-lite-llama-eight-gpu-1", created_at=time.time(), workers=workers,
                session="medtoolbench-b-lite-llama-eight-v1")
    plan["resume"] = {"authorization": "User 2026-09-29: switched to eight GPUs; start experiments",
        "control": str(control), "previous_control": str(PREVIOUS), "previous_plan_hash": previous["plan_hash"],
        "only_model": "llama31_8b", "adopted_shards": adopted, "administrative_restarts": interruptions,
        "adopted_formal_episodes": 18, "remaining_formal_episodes": 1782,
        "same_protocol_checkpoint_adoption": True, "different_protocol_rows_pooled": False,
        "old_partial_rows_pooled": False, "entire_interrupted_shards_restarted": True,
        "restart_selected_on_correctness": False, "new_attempts_per_remaining_job": 1,
        "stress_inherited_from": str(stress_path), "stress_inherited_sha256": sha(stress_path),
        "no_development_rerun": True, "prior_stop_incident_preserved": True,
        "durable_stop_control": "PAUSE_REQUESTED.json; eligible and attempted dispatch both check this marker"}
    for path in (Path(__file__), ROOT / "repo/tests/test_b_lite_eight.py"):
        plan["source_hashes"][str(path)] = digest(path.read_text())
    for path in (PREVIOUS / "plan.json", HANDOFF / "ready_for_eight_gpu_handoff.json",
                 HANDOFF / "evidence_hashes.json", stress_path, control / "legacy_inventory.json"):
        plan["input_hashes"][str(path)] = digest(path.read_bytes().hex())
    assert_science(plan, previous)
    plan["plan_hash"] = digest(plan)
    write_json(control / "plan.json", plan)
    for job_id, reference in adopted.items():
        folder = control / "jobs" / job_id
        folder.mkdir(parents=True)
        old_result = read(reference["result"])
        write_json(folder / "result.json", {"job": job_id, "model": "llama31_8b", "stage": reference["stage"],
            "run": reference["run"], "worker": old_result["worker"], "state": "complete",
            "episodes": reference["episodes"], "plan_hash": plan["plan_hash"],
            "adopted_from_plan_hash": previous["plan_hash"], "original_result": reference["result"],
            "inference_rerun": False, "time": time.time()})
    folder = control / "preflight/llama31_8b"
    folder.mkdir(parents=True)
    inherited = {"all_pass": True, "plan_hash": plan["plan_hash"], "model": "llama31_8b",
        "gate_type": "identical-hardware/backend/batch stress certificate adoption, not a new measurement",
        "original_receipt": str(stress_path), "original_sha256": sha(stress_path),
        "original_plan_hash": previous["plan_hash"], "new_stress_run": False, "time": time.time()}
    write_json(folder / "complete.json", inherited)
    write_json(folder / "result.json", {**inherited, "state": "complete", "stage": "preflight",
                                       "job": "preflight--llama31_8b", "episodes": 0})
    write_json(control / "matrix.json", {"plan_hash": plan["plan_hash"], "workers": workers,
        "formal_total": 1800, "adopted_formal": 18, "remaining_formal": 1782, "development_adopted": 144,
        "interrupted_shards_restarted": list(interruptions), "partial_formal_excluded": 36,
        "same_scientific_assignment": True, "no_other_model_dispatch": True})
    load(control)
    print(json.dumps({"control": str(control), "plan_hash": plan["plan_hash"],
                      "remaining_formal": 1782, "workers": len(workers)}), flush=True)


def verify_completion(output, plan, job):
    reference = plan["resume"]["adopted_shards"].get(job["id"])
    if reference:
        if output.resolve() != Path(reference["run"]).resolve():
            raise RuntimeError("Adopted checkpoint output path changed")
        prior = read(PREVIOUS / "plan.json")
        old_job = next(j for j in prior["jobs"] if j["id"] == job["id"])
        if job != old_job or reference["previous_plan_hash"] != prior["plan_hash"]:
            raise RuntimeError("Adopted scientific assignment differs")
        if any(not Path(p).is_file() or sha(p) != h for p, h in reference["files"].items()):
            raise RuntimeError("Imported checkpoint changed")
        return OLD_VERIFY(output, prior, old_job)
    expected = Path(plan["resume"]["control"]) / "jobs" / job["id"] / "attempt1"
    if output.resolve() != expected.resolve():
        raise RuntimeError("Unregistered checkpoint; partial old rows cannot be imported")
    return OLD_VERIFY(output, plan, job)


def drain_requested(control, plan):
    marker = read(Path(control) / "PAUSE_REQUESTED.json")
    if marker is None:
        return False
    if marker.get("plan_hash") != plan["plan_hash"] or marker.get("action") != "drain":
        raise RuntimeError("Invalid pause control; fail closed")
    return True


def eligible(control, plan, active, blocked):
    return [] if drain_requested(control, plan) else OLD_ELIGIBLE(control, plan, active, blocked)


def attempt(control, job, worker, owner_fd, stop):
    plan = load(control)
    if drain_requested(control, plan):
        return {"state": "not_dispatched", "job": job["id"], "model": job["model"], "stage": job["stage"]}
    if job["id"] in plan["resume"]["adopted_shards"] or job["stage"] != "test":
        raise RuntimeError("Completed/development evidence cannot be dispatched again")
    return OLD_ATTEMPT(control, job, worker, owner_fd, stop)


def run_shard(control, job_id, worker_name, output):
    plan = load(control)
    worker = validate_worker(plan, worker_name)
    if job_id in plan["resume"]["adopted_shards"]:
        raise RuntimeError("Cannot rerun an adopted checkpoint")
    from .b_lite_transport import JsonModel

    class BoundLlamaModel(JsonModel):
        def __init__(self, *args, **kwargs):
            import torch
            if torch.cuda.device_count() != 1 or not same_device(torch.cuda.get_device_properties(0).uuid, worker["uuid"]):
                raise RuntimeError("CUDA device is not the exact physical GPU assigned to this lane")
            super().__init__(*args, **kwargs)
            self.metadata["resource_resume"] = {"plan_hash": plan["plan_hash"], "worker": worker,
                "cuda_uuid": str(torch.cuda.get_device_properties(0).uuid), "scientific_adapter_unchanged": True}

    four.model_class = lambda system: BoundLlamaModel
    OLD_RUN_SHARD(control, job_id, worker_name, output)


def install():
    four.CONTROL, four.MODULE, four.load = CONTROL, MODULE, load
    four.verify_completion = verify_completion
    four.base.MODELS, four.base.CONTROL = ("llama31_8b",), CONTROL
    four.base.load, four.base.attempt, four.base.verify_completion = load, attempt, verify_completion
    four.base.eligible = eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "run", "shard", "report", "drain"))
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--job")
    parser.add_argument("--worker")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.control)
        return
    install()
    if args.action == "validate":
        four.base.validate(args.control)
    elif args.action == "shard":
        run_shard(args.control, args.job, args.worker, args.out)
    elif args.action == "report":
        four.base.report(args.control, load(args.control))
    elif args.action == "drain":
        plan = load(args.control)
        if (args.control / "PAUSE_REQUESTED.json").exists():
            raise RuntimeError("Drain is already requested")
        write_json(args.control / "PAUSE_REQUESTED.json", {"action": "drain", "plan_hash": plan["plan_hash"],
            "time": time.time(), "instruction": "No new dispatch; finish in-flight shards; no automatic resumption"})
    else:
        try:
            four.base.run(args.control, args.resume)
        except BaseException:
            write_json(args.control / "controller_failure.json", {"error": traceback.format_exc(), "time": time.time()})
            raise


if __name__ == "__main__":
    main()
