"""Lease-reserved continuation at natural job boundaries, with bounded OOM repair.

Existing supervisors remain alive. This controller reserves every unstarted job
using their SAME job locks, waits for an old worker to become idle, then runs
unchanged jobs on that GPU. Live model processes are never signalled. Reservations
are inherited by every child, so a controller crash cannot expose pending jobs
to legacy workers while a repair child still owns a GPU.
"""
import argparse
import json
import os
import shlex
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import FileLease, validate_worker
from medtoolbench.research.cluster_supervisor import execute
from medtoolbench.research.job_state import write_json
from .campaign import gate
from .pipeline import load as load_pipeline
from .runner import load_plan


def load(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    checksum = manifest.pop("manifest_hash")
    if digest(manifest) != checksum:
        raise RuntimeError("Memory execution manifest changed")
    manifest["manifest_hash"] = checksum
    for name, value in manifest["source_hashes"].items():
        if digest(Path(name).read_text()) != value:
            raise RuntimeError("Memory execution source changed: " + name)
    base = load_pipeline(Path(manifest["canonical_pipeline"]))
    if (base["pipeline_hash"] != manifest["canonical_pipeline_hash"] or
            base["campaigns"] != manifest["campaigns"] or base["plan_hashes"] != manifest["plan_hashes"]):
        raise RuntimeError("Scientific plans changed under memory addendum")
    if manifest["chunk_tokens"] != 512 or manifest["extra_infrastructure_attempts"] != 2:
        raise RuntimeError("Undeclared memory or retry policy")
    return manifest


def read_result(folder):
    path = folder / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def authorize_oom_repair(folder, job):
    """No re-selection on accuracy, audit failure, or missing process evidence."""
    result = read_result(folder)
    if job["model"] != "gemma3_12b" or not result or result.get("state") != "failed":
        raise RuntimeError("Only terminal failed Gemma jobs qualify for OOM repair")
    attempts = sorted(p for p in folder.glob("attempt*") if p.is_dir() and p.name[7:].isdigit())
    if not attempts:
        raise RuntimeError("Missing failure evidence")
    failures = []
    for attempt in attempts:
        if (attempt / "audit.json").exists() or (attempt / "complete.json").exists():
            raise RuntimeError("Never retry audited/completed evidence based on outcomes")
        path = attempt / "failure.json"
        if not path.exists():
            raise RuntimeError("Unresolved earlier attempt")
        failure = json.loads(path.read_text())
        if "torch.OutOfMemoryError: CUDA out of memory" not in failure.get("error", ""):
            raise RuntimeError("Non-OOM failure needs separate diagnosis")
        failures.append({"attempt": str(attempt), "failure_hash": digest(failure)})
    return {"reason": "CUDA OOM only; outcome-independent infrastructure repair", "prior_result": result,
            "prior_result_hash": digest(result), "prior_failures": failures, "prior_partial_rows_pooled": False}


def verify_completion(out, plan, job):
    completion = json.loads((out / "complete.json").read_text())
    proof = json.loads((out / "audit.json").read_text())
    config = json.loads((out / "config.json").read_text())
    expected = {e["key"] for e in job["episodes"]}
    if not completion["all_pass"] or not proof["all_pass"] or proof["episodes"] != len(expected):
        raise RuntimeError("Independent audit/count failed; no outcome-based rerun")
    if proof["plan_hash"] != plan["plan_hash"] or config["plan_hash"] != plan["plan_hash"] or config["job"] != job:
        raise RuntimeError("Attempt identity mismatch")
    if proof.get("missing") or proof.get("unexpected") or proof.get("errors"):
        raise RuntimeError("Incomplete audit coverage")
    return proof


def attempt_job(manifest_path, plan_path, job, row, inherited, control, development=False):
    manifest = load(manifest_path)
    plan = load_plan(plan_path)
    folder = (control / "qualification" / job["id"]) if development else (plan_path.parent / "jobs" / job["id"])
    folder.mkdir(parents=True, exist_ok=True)
    status = control / "workers" / (row["name"] + ".json")
    previous = sorted(p for p in folder.glob("attempt*") if p.is_dir() and p.name[7:].isdigit())
    if previous and not (folder / "memory_repair_authorization.json").exists():
        raise RuntimeError("Unexpected prior attempts; refuse silent restart")
    first = max((int(p.name[7:]) for p in previous), default=0) + 1
    attempts = []
    for number in range(first, first + manifest["extra_infrastructure_attempts"]):
        load(manifest_path)
        out = folder / ("attempt" + str(number))
        module = "medtoolbench.supplement.memory_runner" if job["model"] == "gemma3_12b" else plan["runner_module"]
        cmd = [plan["python"], "-u", "-m", module]
        if job["model"] == "gemma3_12b":
            cmd += ["--memory-manifest", str(manifest_path)]
            if development:
                cmd += ["--development-validation"]
        cmd += ["--plan", str(plan_path), "--worker", row["name"], "--job", job["id"], "--out", str(out)]
        write_json(folder / ("command" + str(number) + ".json"), {"command": cmd, "worker": row,
            "execution_manifest": str(manifest_path), "execution_manifest_hash": manifest["manifest_hash"]})
        execution = execute(cmd, folder / ("attempt" + str(number) + ".log"), status, row, inherited,
                            stall_seconds=3600, wall_seconds=64800)
        attempts.append({"attempt": number, **execution})
        if (out / "complete.json").exists() and execution["returncode"] == 0:
            proof = verify_completion(out, plan, job)
            record = {"state": "complete", "job": job["id"], "model": job["model"], "episodes": proof["episodes"],
                      "run": str(out), "attempts": attempts, "worker": row["name"],
                      "execution_manifest": str(manifest_path), "execution_manifest_hash": manifest["manifest_hash"]}
            if (folder / "memory_repair_authorization.json").exists():
                record["prior_failure_record"] = str(folder / "result.before_memory_repair.json")
            write_json(folder / "result.json", record)
            write_json(status, {"state": "idle", "worker": row, "heartbeat": time.time(), "finished_job": job["id"]})
            return record
        if (out / "audit.json").exists() or (out / "complete.json").exists():
            raise RuntimeError("Audit-invalid evidence is not retried")
    write_json(folder / "result.json", {"state": "failed", "job": job["id"], "model": job["model"],
        "attempts": attempts, "execution_manifest_hash": manifest["manifest_hash"],
        "error": "Memory execution infrastructure retry budget exhausted"})
    raise RuntimeError("Memory execution infrastructure retry budget exhausted: " + job["id"])


def launch(path):
    manifest = load(path)
    plan = load_plan(Path(manifest["campaigns"][0]))
    for row in plan["workers"]:
        validate_worker(plan, row["name"])
    session = manifest["session"]
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError("Recovery session already exists")
    root = plan["project"]
    cmd = ["env", "PYTHONPATH=" + str(Path(root) / "repo"), plan["python"], "-u", "-m",
           "medtoolbench.supplement.memory_execution", "--manifest", str(path)]
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "supervisor", "-c", root,
                    "exec " + shlex.join(cmd)], check=True)
    write_json(path.parent / "launch.json", {"manifest_hash": manifest["manifest_hash"], "time": time.time(),
        "session": session, "existing_model_processes_interrupted": False})


def run(path):
    manifest = load(path)
    control = path.parent
    plans = [(Path(p), load_plan(Path(p))) for p in manifest["campaigns"]]
    rows = plans[0][1]["workers"]
    for row in rows:
        validate_worker(plans[0][1], row["name"])
    leases, adopted, busy, completed_futures = {}, set(), {}, []
    all_jobs = {(str(cp), j["id"]): (cp, plan, j) for cp, plan in plans for j in plan["jobs"]}
    (control / "workers").mkdir(exist_ok=True)
    (control / "handoff").mkdir(exist_ok=True)

    def reserve():
        for key, (cp, plan, job) in all_jobs.items():
            if key in leases:
                continue
            folder = cp.parent / "jobs" / job["id"]
            result = read_result(folder)
            if result and result["state"] == "complete":
                continue
            lease = FileLease(cp.parent / "locks" / (job["id"] + ".lock"))
            try:
                lease.__enter__()
            except BlockingIOError:
                continue  # An existing live attempt owns this job; never touch it.
            leases[key] = lease
            result = read_result(folder)
            if result and result["state"] != "complete":
                authorization = authorize_oom_repair(folder, job)
                if (folder / "memory_repair_authorization.json").exists():
                    raise RuntimeError("Repair already attempted; bounded retry policy")
                write_json(folder / "memory_repair_authorization.json", {**authorization,
                    "manifest_hash": manifest["manifest_hash"], "time": time.time()})
                # Preserve the exact old terminal record rather than erase it.
                (folder / "result.json").rename(folder / "result.before_memory_repair.json")
            elif not result and list(folder.glob("attempt[0-9]*")):
                raise RuntimeError("Unowned unresolved attempt requires explicit review: " + str(folder))

    def adopt_idle():
        for row in rows:
            if row["name"] in adopted:
                continue
            p = plans[0][0].parent / "workers" / (row["name"] + ".json")
            if not p.exists():
                continue
            old = json.loads(p.read_text())
            # All remaining tasks are now reserved, so a legacy waiting worker
            # cannot start new inference. The child still enforces the GPU lease.
            if old["state"] in {"waiting", "complete", "failed"} and old["worker"] == row:
                adopted.add(row["name"])
                write_json(control / "handoff" / (row["name"] + ".json"), {"worker": row, "time": time.time(),
                    "old_state": old, "method": "Natural attempt boundary with all pending jobs reserved",
                    "model_process_interrupted": False})

    def fds(owner):
        return [owner.fileno()] + [lease.fileno() for lease in leases.values()]

    try:
        with FileLease(control / "supervisor.lock") as owner, ThreadPoolExecutor(max_workers=8) as pool:
            reserve()
            proof_future = None
            while True:
                reserve()
                adopt_idle()
                for name, (future, key) in list(busy.items()):
                    if future.done():
                        future.result()  # Fail closed, retaining all evidence.
                        completed_futures.append(key)
                        del busy[name]
                proof_path = control / "preflight/complete.json"
                free = [r for r in rows if r["name"] in adopted and r["name"] not in busy]
                if not proof_path.exists():
                    if proof_future is None and free:
                        row = free[0]
                        out = control / "preflight"
                        cmd = [plans[0][1]["python"], "-u", "-m", "medtoolbench.supplement.memory_preflight",
                               "--manifest", str(path), "--worker", row["name"], "--out", str(out)]
                        proof_future = pool.submit(execute, cmd, control / "preflight.log", control / "workers" / (row["name"] + ".json"),
                                                   row, fds(owner), 3600, 14400)
                        busy[row["name"]] = (proof_future, "preflight")
                    if proof_future is not None and proof_future.done():
                        result = proof_future.result()
                        if result["returncode"] != 0 or not proof_path.exists():
                            raise RuntimeError("Real-GPU memory preflight failed; no adapted test jobs were launched")
                    write_json(control / "status.json", {"state": "waiting_for_memory_preflight", "reserved_jobs": len(leases),
                        "adopted_workers": sorted(adopted), "busy": {n: str(k) for n, (_, k) in busy.items()}, "heartbeat": time.time()})
                    time.sleep(5)
                    continue
                proof = json.loads(proof_path.read_text())
                if not proof["all_pass"] or proof["manifest_hash"] != manifest["manifest_hash"]:
                    raise RuntimeError("Invalid memory gate")
                dev_root = control / "qualification"
                if not (dev_root / "complete.json").exists():
                    cp, plan = plans[0]
                    dev_jobs = [j for j in plan["jobs"] if j["stage"] == "qualification" and j["model"] == "gemma3_12b"]
                    dev_results = [read_result(dev_root / j["id"]) for j in dev_jobs]
                    if all(r and r["state"] == "complete" for r in dev_results):
                        write_json(dev_root / "complete.json", {"all_pass": True, "manifest_hash": manifest["manifest_hash"],
                            "jobs": len(dev_jobs), "episodes": sum(r["episodes"] for r in dev_results),
                            "held_out_episodes": 0, "original_qualification_evidence_preserved": True, "time": time.time()})
                    else:
                        active_keys = {key for _, key in busy.values()}
                        pending = [j for j in dev_jobs if not read_result(dev_root / j["id"])
                                   and ("qualification", j["id"]) not in active_keys]
                        for row, job in zip(free, pending):
                            key = ("qualification", job["id"])
                            busy[row["name"]] = (pool.submit(attempt_job, path, cp, job, row, fds(owner), control, True), key)
                        write_json(control / "status.json", {"state": "memory_development_regression", "reserved_jobs": len(leases),
                            "adopted_workers": sorted(adopted), "busy": {n: str(k) for n, (_, k) in busy.items()}, "heartbeat": time.time()})
                        time.sleep(5)
                    continue
                stage = None
                for index in manifest["execution_order"]:
                    cp, plan = plans[index]
                    results = [read_result(cp.parent / "jobs" / j["id"]) for j in plan["jobs"]]
                    if not all(r and r["state"] == "complete" for r in results):
                        stage = (cp, plan)
                        break
                if stage is None:
                    write_json(control / "inference_complete.json", {"all_pass": True, "manifest_hash": manifest["manifest_hash"],
                        "episodes_including_qualification": sum(len(j["episodes"]) for _, p in plans for j in p["jobs"]),
                        "time": time.time(), "statistical_reports_complete": False})
                    break
                cp, plan = stage
                gate_state = gate(cp.parent, plan)
                if gate_state == "failed":
                    raise RuntimeError("Scientific qualification/oracle gate failed: " + str(cp))
                active_keys = {key for _, key in busy.values()}
                candidates = [j for j in plan["jobs"] if (str(cp), j["id"]) in leases
                    and (str(cp), j["id"]) not in active_keys and not read_result(cp.parent / "jobs" / j["id"])
                    and (j["stage"] == "qualification" or gate_state == "passed")]
                # Fix pre-existing OOM holes first; no ordering by model accuracy.
                candidates.sort(key=lambda j: not (cp.parent / "jobs" / j["id"] / "memory_repair_authorization.json").exists())
                for row, job in zip(free, candidates):
                    key = (str(cp), job["id"])
                    future = pool.submit(attempt_job, path, cp, job, row, fds(owner), control)
                    busy[row["name"]] = (future, key)
                write_json(control / "status.json", {"state": "running", "campaign": str(cp), "gate": gate_state,
                    "reserved_jobs": len(leases), "adopted_workers": sorted(adopted),
                    "busy": {n: str(k) for n, (_, k) in busy.items()}, "heartbeat": time.time()})
                time.sleep(5)
        # Original controllers still produce campaign audits/reports. A watchdog
        # here also finalizes safely under their existing reporting locks.
        from .report import generate
        for cp, plan in plans:
            while not (cp.parent / "report/complete.json").exists():
                try:
                    with FileLease(cp.parent / "locks/statistical-report.lock"):
                        if not (cp.parent / "report/complete.json").exists():
                            generate(cp)
                except BlockingIOError:
                    time.sleep(15)
        write_json(control / "complete.json", {"all_pass": True, "manifest_hash": manifest["manifest_hash"],
            "time": time.time(), "campaigns": manifest["campaigns"], "historical_failures_preserved": True})
    except Exception:
        write_json(control / "failed.json", {"error": traceback.format_exc(), "time": time.time(),
            "manifest_hash": manifest["manifest_hash"], "complete": False, "pending_job_reservations_retained": True})
        # Do not silently release the queue back to the known-OOM implementation
        # if validation or recovery fails. Retain leases until explicit review.
        while True:
            write_json(control / "status.json", {"state": "failed_closed", "heartbeat": time.time(),
                "reserved_jobs": len(leases), "failure": str(control / "failed.json")})
            time.sleep(15)
    finally:
        for lease in leases.values():
            lease.__exit__()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--launch", action="store_true")
    a = p.parse_args()
    (launch if a.launch else run)(a.manifest.resolve())


if __name__ == "__main__":
    main()
