"""Record the requested Qwen-only finish/pause without changing frozen runners.

The existing controller naturally exits when Qwen finishes because Llama is
blocked. This CPU-only observer sends no signals and launches no model jobs.
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

from medtoolbench.supplement.b_lite import ROOT, CONTROL, read, sha
from medtoolbench.supplement.b_lite_execution import load
from medtoolbench.supplement.memory_execution import verify_completion
from medtoolbench.research.cluster_execution import FileLease
from medtoolbench.research.job_state import write_json

PAUSE = ROOT / "data/b_lite_pause_after_qwen_v1"
MODEL = "qwen35_9b"


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        return {"pid": pid, "start_ticks": fields[19], "command": command, "state": fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(wanted, current):
    return bool(wanted and current and current["state"] != "Z" and
                all(wanted[k] == current[k] for k in ("pid", "start_ticks", "command")))


def assess(plan, control):
    counts, pending, failed, done = 0, [], [], []
    for job in plan["jobs"]:
        folder = control / "jobs" / job["id"]
        if job["model"] != MODEL:
            if job["stage"] == "test" and folder.exists():
                raise RuntimeError("Unexpected non-Qwen formal work; pause scope requires review")
            continue
        result = read(folder / "result.json")
        if not result:
            pending.append(job["id"])
        elif result.get("state") != "complete":
            failed.append(job["id"])
        else:
            done.append((job, result))
            if job["stage"] == "test":
                counts += len(job["episodes"])
    expected = sum(len(j["episodes"]) for j in plan["jobs"]
                   if j["model"] == MODEL and j["stage"] == "test")
    return {"formal_expected": expected, "audited_formal": counts,
            "pending_jobs": pending, "failed_jobs": failed,
            "qwen_complete": counts == expected and not pending and not failed}, done


def arm(pause):
    plan = load(CONTROL)
    gate = read(CONTROL / "gates/llama31_8b.json")
    status = read(CONTROL / "status.json")
    if not gate or gate.get("all_pass") or status.get("blocked_models", {}).get("llama31_8b") is None:
        raise RuntimeError("Natural Qwen-only exit is not established")
    if any(r["model"] != MODEL for r in status["active"]):
        raise RuntimeError("Unexpected non-Qwen active work")
    started = read(CONTROL / "started.json")
    parent = identity(started["pid"])
    state, _ = assess(plan, CONTROL)
    if parent and ("medtoolbench.supplement.b_lite_execution run" not in parent["command"]):
        raise RuntimeError("Original controller PID was reused")
    if not parent and not state["qwen_complete"]:
        raise RuntimeError("Qwen incomplete and controller not visible; cannot arm in isolated PID namespace")
    pause.mkdir(parents=True, exist_ok=False)
    request = {"time": time.time(), "instruction": "finish and audit Qwen, then pause until user switches to four GPUs",
        "plan": str(CONTROL / "plan.json"), "plan_hash": plan["plan_hash"],
        "controller": parent, "model": MODEL, "new_model_launch_authorized": False,
        "auto_resume_authorized": False, "future_gpu_count": 4,
        "stop_mechanism": "existing controller natural exit with Llama gate blocked",
        "observer_sends_signals": False, "frozen_sources_modified": False,
        "watcher_source_sha256": sha(Path(__file__)),
        "llama_gate_sha256": sha(CONTROL / "gates/llama31_8b.json"), "initial_progress": state}
    write_json(pause / "request.json", request)
    print(json.dumps(request, indent=2), flush=True)


def certify(pause, request, plan, done):
    report = read(CONTROL / "report/metrics.json")
    if not report or report["plan_hash"] != plan["plan_hash"] or not report["models"][MODEL]["complete"]:
        raise RuntimeError("Qwen complete report not available after controller exit")
    # Prove the supervisor and each job have actually relinquished ownership.
    with FileLease(CONTROL / "supervisor.lock"):
        files = {}
        for job, result in done:
            with FileLease(CONTROL / "locks" / (job["id"] + ".lock")):
                output = Path(result["run"])
                verify_completion(output, plan, job)
                wanted = {e["key"] + ".json.gz" for e in job["episodes"]}
                if {p.name for p in (output / "episodes").glob("*.json.gz")} != wanted:
                    raise RuntimeError("Saved episode file coverage mismatch")
                paths = [output.parent / "result.json", output / "config.json", output / "audit.json",
                         output / "complete.json", *sorted((output / "episodes").glob("*.json.gz"))]
                files.update({str(p): sha(p) for p in paths})
        for p in (CONTROL / "plan.json", CONTROL / "selection.json", CONTROL / "unit.json",
                  CONTROL / "report/metrics.json", CONTROL / "report/RESULTS.md"):
            files[str(p)] = sha(p)
        write_json(pause / "qwen_evidence_hashes.json", {"sha256": files, "time": time.time()})
        write_json(pause / "ready_for_four_gpu_handoff.json", {
            "all_pass": True, "time": time.time(), "model": MODEL, "formal_episodes": 1800,
            "development_episodes": 144, "plan_hash": plan["plan_hash"],
            "qwen_complete": True, "original_controller_exited": True, "qwen_job_locks_released": True,
            "automatic_resumption": False, "new_model_started": False,
            "entire_b_lite_complete": False, "llama_formal_episodes": 0,
            "future_gpu_count": 4, "future_resource_manifest_required": True,
            "future_llama_protocol_validation_required": True,
            "evidence_inventory": str(pause / "qwen_evidence_hashes.json"),
            "note": "Do not resume the frozen eight-GPU manifest on four GPUs. Await user handoff."})


def watch(pause):
    request = read(pause / "request.json")
    if request["watcher_source_sha256"] != sha(Path(__file__)):
        raise RuntimeError("Observer changed since request")
    plan = load(CONTROL)
    if plan["plan_hash"] != request["plan_hash"]:
        raise RuntimeError("Frozen plan identity mismatch")
    with FileLease(pause / "observer.lock"):
        write_json(pause / "observer_started.json", {"pid": os.getpid(), "time": time.time()})
        while True:
            if sha(CONTROL / "gates/llama31_8b.json") != request["llama_gate_sha256"]:
                raise RuntimeError("Llama gate changed; review requested pause scope")
            state, done = assess(plan, CONTROL)
            parent = request["controller"]
            alive = same_process(parent, identity(parent["pid"])) if parent else False
            state.update(heartbeat=time.time(), controller_alive=alive, state="waiting_for_qwen_and_natural_exit")
            write_json(pause / "status.json", state)
            if not alive:
                if not state["qwen_complete"]:
                    raise RuntimeError("Controller exited before Qwen completion; no successful handoff claimed")
                load(CONTROL)
                certify(pause, request, plan, done)
                state.update(state="paused_ready_for_four_gpus", heartbeat=time.time())
                write_json(pause / "status.json", state)
                return
            time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("arm", "watch"))
    parser.add_argument("--pause", type=Path, default=PAUSE)
    args = parser.parse_args()
    if args.action == "arm":
        arm(args.pause)
    else:
        try:
            watch(args.pause)
        except BaseException:
            write_json(args.pause / "observer_failure.json", {"error": traceback.format_exc(), "time": time.time()})
            raise
