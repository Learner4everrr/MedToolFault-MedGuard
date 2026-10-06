"""Drain Llama development only, then stop for an explicit eight-GPU handoff.

SIGSTOP targets only the existing dispatcher, never its running child. Once
every recorded child has exited, kill the stopped dispatcher without resuming
it, reconcile independently audited checkpoints, and certify released leases.
No model process is launched; no frozen scientific file or plan is edited.
"""
import argparse
import json
import os
import signal
import time
import traceback
from pathlib import Path

from medtoolbench.research.cluster_execution import FileLease, AllocatedGpuLease, validate_worker
from medtoolbench.research.job_state import write_json
from . import b_lite_four as four
from .b_lite import ROOT, read, sha

CONTROL = ROOT / "data/b_lite_llama_four_gpu_v2"
PAUSE = ROOT / "data/b_lite_pause_after_llama_dev_v1"
RUNNER = "medtoolbench.supplement.b_lite_four_identity"


def identity(pid):
    try:
        root = Path("/proc") / str(pid)
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        args = (root / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        return {"pid": pid, "start_ticks": fields[19], "state": fields[0],
                "ppid": int(fields[1]), "args": args,
                "exit_status": int(fields[49]) if fields[0] == "Z" else None}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(wanted, current):
    return bool(wanted and current and all(wanted[k] == current[k] for k in ("pid", "start_ticks")))


def no_formal(plan, control):
    if any((control / "jobs" / j["id"]).exists() for j in plan["jobs"] if j["stage"] == "test"):
        raise RuntimeError("Formal dispatch detected; no zero-formal handoff can be claimed")


def development(plan):
    jobs = [j for j in plan["jobs"] if j["stage"] == "qualification"]
    if len(jobs) != 4 or any(j["model"] != "llama31_8b" for j in jobs):
        raise RuntimeError("Unexpected development scope")
    return jobs


def arm(control, pause):
    plan = four.load(control)
    no_formal(plan, control)
    status = read(control / "status.json")
    parent = identity(read(control / "started.json")["pid"])
    if not parent or parent["args"][-3:] != ["-m", RUNNER, "run"]:
        raise RuntimeError("Verified original dispatcher required")
    if any(r["stage"] != "qualification" or r["model"] != "llama31_8b" for r in status["active"]):
        raise RuntimeError("Unexpected active work")
    if parent["state"] not in {"T", "t"}:
        os.kill(parent["pid"], signal.SIGSTOP)
    # Stabilize the dispatch fence before recording children; never send CONT.
    for _ in range(20):
        current = identity(parent["pid"])
        if same_process(parent, current) and current["state"] in {"T", "t"}:
            break
        time.sleep(.1)
    else:
        raise RuntimeError("Dispatcher did not enter stopped state")
    no_formal(plan, control)
    recorded, completed_hashes = [], {}
    for job in development(plan):
        folder = control / "jobs" / job["id"]
        result = read(folder / "result.json")
        if result:
            if result["state"] != "complete":
                raise RuntimeError("Existing failed development requires separate review")
            four.verify_completion(Path(result["run"]), plan, job)
            for p in folder.rglob("*.json*"):
                if p.is_file():
                    completed_hashes[str(p)] = sha(p)
            continue
        matches = []
        for p in (control / "workers").glob("*.json"):
            worker = read(p)
            if worker.get("state") == "running" and worker.get("out") == str(folder / "attempt1"):
                matches.append(worker)
        if len(matches) != 1:
            raise RuntimeError("Cannot identify unique running development child")
        worker = matches[0]
        child = identity(worker["pid"])
        if not child or child["ppid"] != parent["pid"]:
            raise RuntimeError("Worker PID is not the original dispatcher's child")
        if child["state"] != "Z" and (RUNNER not in child["args"] or job["id"] not in child["args"]):
            raise RuntimeError("Worker command identity mismatch")
        recorded.append({"job": job["id"], "process": child, "worker": worker["worker"],
                         "output": str(folder / "attempt1")})
    pause.mkdir(parents=True, exist_ok=False)
    request = {"instruction": "Finish and audit Llama development, then stop; await user switch to eight GPUs",
        "time": time.time(), "plan_hash": plan["plan_hash"], "control": str(control),
        "controller": parent, "children": recorded, "initial_status": status,
        "initial_completed_evidence": completed_hashes, "observer_sha256": sha(Path(__file__)),
        "formal_launch_authorized": False, "automatic_resumption": False, "future_gpu_count": 8,
        "mechanism": "Dispatcher SIGSTOP only; completed child receipts reconciled after dispatcher termination",
        "running_child_interrupted": False, "frozen_sources_modified": False}
    write_json(pause / "request.json", request)
    print(json.dumps({"armed": True, "pause": str(pause), "controller": parent["pid"],
                      "children": [c["process"]["pid"] for c in recorded], "formal_started": False}), flush=True)


def exit_receipt(record):
    current = identity(record["process"]["pid"])
    if not same_process(record["process"], current):
        raise RuntimeError("Child disappeared or PID changed before exit status was recorded")
    if current["state"] != "Z":
        return None
    return {"job": record["job"], "process": current,
            "returncode": os.waitstatus_to_exitcode(current["exit_status"]), "observed_at": time.time()}


def finish(control, pause, request, plan, exits):
    no_formal(plan, control)
    if len(exits) != len(request["children"]) or any(e is None for e in exits):
        raise RuntimeError("Cannot terminate dispatcher before all children exit")
    for child in request["children"]:
        if exit_receipt(child) is None:
            raise RuntimeError("Development child is still running")
    parent = identity(request["controller"]["pid"])
    if not same_process(request["controller"], parent) or parent["state"] not in {"T", "t"}:
        raise RuntimeError("Stopped dispatcher identity changed")
    # All children are zombies: they have finished inference, auditing and cleanup.
    # Never resume a frozen dispatcher, which could race a new formal submission.
    write_json(pause / "child_exit_receipts.json", {"exits": exits, "time": time.time()})
    os.kill(parent["pid"], signal.SIGKILL)
    for _ in range(100):
        now = identity(parent["pid"])
        if not same_process(parent, now) or now["state"] == "Z":
            break
        time.sleep(.1)
    else:
        raise RuntimeError("Dispatcher termination was not confirmed")
    if any(sha(p) != value for p, value in request["initial_completed_evidence"].items()):
        raise RuntimeError("Previously audited development evidence changed")
    by_exit = {e["job"]: e for e in exits}
    by_child = {c["job"]: c for c in request["children"]}
    with FileLease(control / "supervisor.lock"):
        for worker in plan["workers"]:
            validate_worker(plan, worker["name"])
            previous = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(worker["gpu"])
                with AllocatedGpuLease(worker):
                    pass
            finally:
                if previous is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = previous
        for job in development(plan):
            folder = control / "jobs" / job["id"]
            with FileLease(control / "locks" / (job["id"] + ".lock")):
                result = read(folder / "result.json")
                if result:
                    four.verify_completion(Path(result["run"]), plan, job)
                    continue
                receipt, child = by_exit[job["id"]], by_child[job["id"]]
                output = Path(child["output"])
                if receipt["returncode"] != 0:
                    raise RuntimeError("Development child failed; no successful completion claimed")
                from .external_runner import audit
                if not audit(output, plan, job)["all_pass"]:
                    raise RuntimeError("Independent development audit failed")
                four.verify_completion(output, plan, job)
                write_json(folder / "result.json", {"job": job["id"], "model": job["model"],
                    "stage": job["stage"], "run": str(output), "worker": child["worker"],
                    "execution": {"returncode": 0, "timeout": None,
                        "exit_receipt": str(pause / "child_exit_receipts.json"),
                        "reconciled_after_user_requested_dispatch_pause": True,
                        "inference_rerun": False}, "plan_hash": plan["plan_hash"],
                    "state": "complete", "episodes": len(job["episodes"]), "time": time.time()})
                write_json(control / "workers" / (child["worker"]["name"] + ".json"),
                    {"state": "idle", "job": job["id"], "result": "complete", "heartbeat": time.time()})
        # Evaluate the unchanged capability gate WITHOUT invoking the scheduler.
        four.install()
        gate = four.base.model_gate(control, plan, "llama31_8b")
        no_formal(plan, control)
        files = {}
        for job in development(plan):
            for p in (control / "jobs" / job["id"]).rglob("*.json*"):
                if p.is_file():
                    files[str(p)] = sha(p)
        for p in (control / "plan.json", control / "unit.json", control / "gates/llama31_8b.json"):
            files[str(p)] = sha(p)
        write_json(pause / "development_evidence_hashes.json", {"sha256": files, "time": time.time()})
        state = {"state": "paused_after_development", "heartbeat": time.time(),
            "plan_hash": plan["plan_hash"], "audited_development": 144,
            "audited_formal": 0, "formal_total": 1800, "active": [], "automatic_resumption": False,
            "development_gate": gate, "future_gpu_count": 8, "pause_control": str(pause)}
        write_json(control / "status.json", state)
        write_json(pause / "status.json", state)
        write_json(pause / "ready_for_eight_gpu_handoff.json", {"all_pass": gate == "passed",
            "time": time.time(), "model": "llama31_8b", "plan_hash": plan["plan_hash"],
            "development_episodes": 144, "development_gate": gate, "formal_episodes": 0,
            "controller_exited": True, "development_job_locks_released": True,
            "allocated_gpu_leases_released": True,
            "automatic_resumption": False, "entire_b_lite_complete": False,
            "future_gpu_count": 8, "future_resource_manifest_required": True,
            "evidence_inventory": str(pause / "development_evidence_hashes.json"),
            "note": "Await explicit user handoff; do not resume the four-GPU manifest on eight GPUs"})


def watch(control, pause):
    request = read(pause / "request.json")
    if request["observer_sha256"] != sha(Path(__file__)) or request["control"] != str(control):
        raise RuntimeError("Pause request/source changed")
    plan = four.load(control)
    if request["plan_hash"] != plan["plan_hash"]:
        raise RuntimeError("Pause plan identity mismatch")
    with FileLease(pause / "observer.lock"):
        write_json(pause / "observer_started.json", {"pid": os.getpid(), "time": time.time()})
        while True:
            no_formal(plan, control)
            parent = identity(request["controller"]["pid"])
            if not same_process(request["controller"], parent):
                raise RuntimeError("Dispatcher disappeared unexpectedly; manual reconciliation required")
            if parent["state"] not in {"T", "t"}:
                os.kill(parent["pid"], signal.SIGSTOP)
                raise RuntimeError("Dispatcher was externally resumed; stopped again for review")
            exits = [exit_receipt(child) for child in request["children"]]
            if all(e is not None for e in exits):
                finish(control, pause, request, plan, exits)
                return
            active, saved = [], 0
            for job in development(plan):
                output = control / "jobs" / job["id"] / "attempt1"
                saved += len(list((output / "episodes").glob("*.json.gz")))
                if job["id"] in {c["job"] for c in request["children"]}:
                    active.append({"job": job["id"], "stage": "qualification", "model": "llama31_8b",
                                   "progress": read(output / "progress.json") or {}})
            audited = sum(len(j["episodes"]) for j in development(plan)
                          if read(control / "jobs" / j["id"] / "result.json"))
            state = {"state": "draining_development_user_pause", "heartbeat": time.time(),
                "plan_hash": plan["plan_hash"], "dispatcher_stopped": True,
                "development_saved": saved, "audited_development": audited,
                "audited_formal": 0, "formal_total": 1800, "active": active,
                "formal_dispatch_disabled": True, "automatic_resumption": False,
                "future_gpu_count": 8, "pause_control": str(pause)}
            write_json(pause / "status.json", state)
            write_json(control / "status.json", state)
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("arm", "watch"))
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--pause", type=Path, default=PAUSE)
    args = parser.parse_args()
    if args.action == "arm":
        arm(args.control, args.pause)
    else:
        try:
            watch(args.control, args.pause)
        except BaseException:
            write_json(args.pause / "observer_failure.json", {"error": traceback.format_exc(), "time": time.time()})
            raise


if __name__ == "__main__":
    main()
