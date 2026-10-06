"""Fault-isolated whole-shard execution with explicit gates and pending ledger.

One new attempt per job in this revision. A failed shard stays in the denominator;
it does not stall other model families or independent campaigns. A model with a
new runtime failure is circuit-broken across campaigns, avoiding repeated OOMs.
Hash/audit failures are integrity errors and still stop dispatch globally.
"""
import argparse
import json
import os
import shlex
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import FileLease, validate_worker
from medtoolbench.research.cluster_supervisor import execute
from medtoolbench.research.job_state import write_json
from .isolated_manifest import load, read, preflight_passes, proof_passes, require_cpu_validation
from .memory_execution import verify_completion
from .runner import load_plan


class IntegrityError(RuntimeError):
    pass


def job_folder(control, cp, job, shadow=False):
    return control / "gemma_development" / job["id"] if shadow else cp.parent / "jobs" / job["id"]


def failure_kind(error):
    if "torch.OutOfMemoryError: CUDA out of memory" in error:
        return "cuda_oom"
    if "state_payload" in error and "'NoneType' object has no attribute 'pop'" in error:
        return "meta_null"
    return "unclassified_infrastructure"


def validate_retry_history(folder):
    """Repair known infrastructure failures only, never bad scientific outcomes."""
    result = read(folder / "result.json")
    attempts = sorted((p for p in folder.glob("attempt*") if p.is_dir() and p.name[7:].isdigit()),
                      key=lambda p: int(p.name[7:]))
    if not attempts and not result:
        return {"prior_attempts": [], "prior_result": None}
    if not result or result.get("state") != "failed" or not attempts:
        raise IntegrityError("Unresolved or successful evidence cannot be restarted")
    history = []
    for attempt in attempts:
        if (attempt / "audit.json").exists() or (attempt / "complete.json").exists():
            raise IntegrityError("Never retry completed/audit-invalid evidence based on outcomes")
        failure = read(attempt / "failure.json")
        if not failure:
            raise IntegrityError("Prior attempt lacks terminal failure evidence")
        kind = failure_kind(failure.get("error", ""))
        if kind not in {"cuda_oom", "meta_null"}:
            raise IntegrityError("Prior failure is outside this repair's declared scope")
        history.append({"path": str(attempt), "failure_hash": digest(failure), "kind": kind})
    return {"prior_attempts": history, "prior_result": result, "prior_result_hash": digest(result)}


def oracle_gate(cp, plan):
    if not plan.get("require_oracle"):
        return "passed"
    root = cp.parent / "oracle"
    if (root / "failed.json").exists():
        return "failed"
    proof = read(root / "complete.json")
    if not proof:
        return "waiting"
    return "passed" if proof.get("all_pass") and proof.get("plan_hash") == plan["plan_hash"] else "failed"


def model_gate(cp, plan, model):
    """No other model's qualification is needed for this model's test jobs."""
    state = oracle_gate(cp, plan)
    if state != "passed":
        return state
    jobs = [j for j in plan["jobs"] if j["stage"] == "qualification" and j["model"] == model]
    if not jobs:
        return "failed"
    results = [read(cp.parent / "jobs" / j["id"] / "result.json") for j in jobs]
    if any(r and r.get("state") != "complete" for r in results):
        return "failed"
    if not all(results):
        return "waiting"
    for job, result in zip(jobs, results):
        verify_completion(Path(result["run"]), plan, job)
    return "passed"


def attempt_job(manifest_path, cp, job, worker, fds, shadow=False):
    manifest = load(manifest_path)
    plan = load_plan(cp)
    control = manifest_path.parent
    folder = job_folder(control, cp, job, shadow)
    folder.mkdir(parents=True, exist_ok=True)
    lock = (control / "locks" / ("shadow-" + job["id"] + ".lock")) if shadow else cp.parent / "locks" / (job["id"] + ".lock")
    with FileLease(lock) as lease:
        history = validate_retry_history(folder)
        marker = folder / "isolated_execution_authorization.json"
        if marker.exists():
            raise IntegrityError("This revision already attempted this job; no unbounded retries")
        number = max((int(Path(r["path"]).name[7:]) for r in history["prior_attempts"]), default=0) + 1
        record = {**history, "manifest_hash": manifest["manifest_hash"], "time": time.time(),
                  "whole_shard_restart": True, "partial_failed_rows_pooled": False}
        write_json(marker, record)
        if history["prior_result"]:
            archive = folder / "result.before_isolated_execution.json"
            if archive.exists():
                raise IntegrityError("History archive already exists")
            # Copy, rather than remove, the old result until a terminal new
            # result is ready; authorization explicitly identifies the new run.
            write_json(archive, history["prior_result"])
        out = folder / ("attempt" + str(number))
        cmd = [plan["python"], "-u", "-m", "medtoolbench.supplement.isolated_runner",
               "--execution-manifest", str(manifest_path)]
        if shadow:
            cmd += ["--shadow-development"]
        cmd += ["--plan", str(cp), "--worker", worker["name"], "--job", job["id"], "--out", str(out)]
        write_json(folder / ("command" + str(number) + ".json"), {"command": cmd, "worker": worker,
            "execution_manifest": str(manifest_path), "execution_manifest_hash": manifest["manifest_hash"]})
        execution = execute(cmd, folder / ("attempt" + str(number) + ".isolated.log"),
            control / "workers" / (worker["name"] + ".json"), worker, [*fds, lease.fileno()],
            stall_seconds=3600, wall_seconds=64800)
        result = {"job": job["id"], "model": job["model"], "worker": worker["name"],
            "run": str(out), "attempts": [{"attempt": number, **execution}],
            "execution_manifest": str(manifest_path), "execution_manifest_hash": manifest["manifest_hash"],
            "shadow_development": shadow, "historical_result_preserved": bool(history["prior_result"])}
        if execution["returncode"] == 0 and (out / "complete.json").exists():
            try:
                proof = verify_completion(out, plan, job)
            except Exception as error:
                raise IntegrityError("Completed run failed identity/coverage audit") from error
            result.update(state="complete", episodes=proof["episodes"])
        else:
            if (out / "audit.json").exists() or (out / "complete.json").exists():
                raise IntegrityError("Audit-invalid evidence requires review, never automatic rerun")
            failure = read(out / "failure.json")
            error = (failure or {}).get("error", "Child terminated without a failure record: " + str(execution))
            result.update(state="quarantined", episodes=0, assigned_episodes=len(job["episodes"]),
                          failure_kind=failure_kind(error), error=error)
        write_json(folder / "result.json", result)
        write_json(control / "workers" / (worker["name"] + ".json"), {"state": "idle",
            "worker": worker, "last_job": job["id"], "last_result": result["state"], "heartbeat": time.time()})
        return result


def eligible_jobs(plans, control, manifest, blocked, active, gemma):
    candidates = []
    gemma_ready = preflight_passes(control, manifest) and proof_passes(control / "gemma_development/complete.json", manifest)
    for index, (cp, plan) in enumerate(plans):
        gates = {}
        for job in plan["jobs"]:
            if (job["model"] == "gemma3_12b") != gemma or job["model"] in blocked:
                continue
            key = (str(cp), job["id"], False)
            if key in active or (gemma and not gemma_ready):
                continue
            folder = cp.parent / "jobs" / job["id"]
            result = read(folder / "result.json")
            if result and result.get("state") == "complete":
                continue
            if (folder / "isolated_execution_authorization.json").exists():
                continue
            if job["stage"] == "qualification":
                # Qualification may run before the oracle, but test never does.
                candidates.append((index, cp, plan, job))
                continue
            if job["model"] not in gates:
                gates[job["model"]] = model_gate(cp, plan, job["model"])
            if gates[job["model"]] == "passed":
                candidates.append((index, cp, plan, job))
    return candidates


def pending_ledger(plans, control):
    pending, done = [], 0
    for cp, plan in plans:
        for job in plan["jobs"]:
            result = read(cp.parent / "jobs" / job["id"] / "result.json")
            if result and result.get("state") == "complete":
                done += len(job["episodes"])
            else:
                pending.append({"campaign": str(cp), "job": job["id"], "stage": job["stage"],
                    "model": job["model"], "assigned_episodes": len(job["episodes"]),
                    "state": result.get("state") if result else "pending",
                    "failure_kind": result.get("failure_kind") if result else None})
    write_json(control / "pending.json", {"jobs": pending, "audited_episodes_including_development": done,
        "pending_episodes_including_development": sum(j["assigned_episodes"] for j in pending),
        "failed_partial_rows_counted": False, "time": time.time()})
    return pending


def run(path):
    manifest = load(path)
    control = path.parent
    impact = require_cpu_validation(control, manifest)
    if not impact["behavior_unchanged_for_saved_trajectories"]:
        raise IntegrityError("Affected completed inputs require explicit harmonization before launch")
    handoff = read(control / "handoff.json")
    if not handoff or not handoff.get("all_legacy_controllers_stopped") or handoff["manifest_hash"] != manifest["manifest_hash"]:
        raise IntegrityError("Safe handoff not recorded")
    if (control / "started.json").exists():
        raise IntegrityError("Controller restart requires explicit reconciliation of child ownership")
    plans = [(Path(p), load_plan(Path(p))) for p in manifest["campaigns"]]
    rows = plans[0][1]["workers"]
    for row in rows:
        validate_worker(plans[0][1], row["name"])
    (control / "workers").mkdir(exist_ok=True)
    (control / "preflight").mkdir(exist_ok=True)
    (control / "gemma_development").mkdir(exist_ok=True)
    busy, blocked, special, cursor, fatal = {}, {}, {}, 0, None
    with ExitStack() as stack:
        owner = stack.enter_context(FileLease(control / "supervisor.lock"))
        leases = [owner]
        for root in manifest["legacy_worker_lock_roots"] + [str(cp.parent) for cp, _ in plans]:
            for row in rows:
                leases.append(stack.enter_context(FileLease(Path(root) / "locks" / (row["name"] + ".lock"))))
        fds = [lease.fileno() for lease in leases]
        write_json(control / "started.json", {"pid": os.getpid(), "manifest_hash": manifest["manifest_hash"], "time": time.time()})
        with ThreadPoolExecutor(max_workers=8) as pool:
            while True:
                try:
                    load(path)
                    for name, (future, key) in list(busy.items()):
                        if not future.done():
                            continue
                        try:
                            result = future.result()
                            if key[0] == "preflight":
                                mode = key[1]
                                passed = result["returncode"] == 0 and proof_passes(control / "preflight" / mode / "complete.json", manifest)
                                special[mode] = "passed" if passed else "failed"
                                if not passed:
                                    blocked["gemma3_12b"] = {"reason": "preflight_" + mode, "execution": result}
                            elif result["state"] != "complete":
                                blocked[result["model"]] = {"reason": result["failure_kind"], "job": result["job"], "run": result["run"]}
                        except Exception:
                            fatal = traceback.format_exc()
                        del busy[name]
                    if fatal:
                        write_json(control / "integrity_failure.json", {"error": fatal, "time": time.time()})
                    free = [r for r in rows if r["name"] not in busy]
                    if not fatal:
                        # Two explicitly reserved validation lanes; all six
                        # other lanes are independent of Gemma's gate/results.
                        for row, mode in zip(rows[-2:], ("compare", "stress")):
                            if row not in free or mode in special:
                                continue
                            out = control / "preflight" / mode
                            cmd = [plans[0][1]["python"], "-u", "-m", "medtoolbench.supplement.isolated_preflight",
                                "--manifest", str(path), "--worker", row["name"], "--mode", mode, "--out", str(out)]
                            busy[row["name"]] = (pool.submit(execute, cmd, control / ("preflight-" + mode + ".log"),
                                control / "workers" / (row["name"] + ".json"), row, fds, 3600, 14400), ("preflight", mode, True))
                            special[mode] = "running"
                        active = {key for _, key in busy.values()}
                        cp, plan = plans[0]
                        dev = [j for j in plan["jobs"] if j["stage"] == "qualification" and j["model"] == "gemma3_12b"]
                        if preflight_passes(control, manifest) and "gemma3_12b" not in blocked:
                            results = [read(job_folder(control, cp, j, True) / "result.json") for j in dev]
                            if all(r and r["state"] == "complete" for r in results):
                                for job, result in zip(dev, results):
                                    verify_completion(Path(result["run"]), plan, job)
                                if not (control / "gemma_development/complete.json").exists():
                                    write_json(control / "gemma_development/complete.json", {
                                        "all_pass": True, "manifest_hash": manifest["manifest_hash"],
                                        "episodes": sum(r["episodes"] for r in results), "held_out_episodes": 0,
                                        "time": time.time()})
                            else:
                                for row in rows[-2:]:
                                    if row["name"] in busy:
                                        continue
                                    ready = [j for j, r in zip(dev, results) if not r
                                        and (str(cp), j["id"], True) not in active]
                                    if ready:
                                        job = ready[0]
                                        key = (str(cp), job["id"], True)
                                        busy[row["name"]] = (pool.submit(attempt_job, path, cp, job, row, fds, True), key)
                                        active.add(key)
                        for row in rows:
                            if row["name"] in busy:
                                continue
                            gemma_lane = row in rows[-2:]
                            candidates = eligible_jobs(plans, control, manifest, blocked, active, gemma_lane)
                            # Once Gemma is blocked or has no remaining work,
                            # lend its lanes to the healthy queue, never idle it
                            # solely because a model-specific repair needs review.
                            if not candidates and gemma_lane and (
                                "gemma3_12b" in blocked or proof_passes(control / "gemma_development/complete.json", manifest)):
                                candidates = eligible_jobs(plans, control, manifest, blocked, active, False)
                            if not candidates and not gemma_lane:
                                candidates = eligible_jobs(plans, control, manifest, blocked, active, True)
                            if not candidates:
                                continue
                            candidates.sort(key=lambda item: (item[3]["stage"] != "qualification",
                                (item[0] - cursor) % len(plans), item[3]["id"]))
                            index, cp, plan, job = candidates[0]
                            cursor = (index + 1) % len(plans)
                            key = (str(cp), job["id"], False)
                            active.add(key)
                            busy[row["name"]] = (pool.submit(attempt_job, path, cp, job, row, fds), key)
                    pending = pending_ledger(plans, control)
                    state = "integrity_stop" if fatal else "running" if busy else "needs_review" if pending else "inference_complete"
                    write_json(control / "status.json", {"state": state, "manifest_hash": manifest["manifest_hash"],
                        "pid": os.getpid(), "heartbeat": time.time(), "busy": {n: list(k) for n, (_, k) in busy.items()},
                        "blocked_models": blocked, "preflight": special, "pending_jobs": len(pending),
                        "all_pending_retained_in_denominator": True})
                    if not busy and (fatal or pending):
                        write_json(control / "needs_review.json", {"blocked_models": blocked,
                            "integrity_error": fatal, "pending_jobs": len(pending), "time": time.time()})
                        return
                    if not pending and not busy:
                        break
                except Exception:
                    fatal = traceback.format_exc()
                time.sleep(5)
        # Existing full-matrix reports are generated only after every job has
        # passed; pending failures never disappear from a partial denominator.
        from .report import generate
        for cp, plan in plans:
            with FileLease(cp.parent / "locks/statistical-report.lock"):
                write_json(cp.parent / "complete.json", {"all_pass": True, "plan_hash": plan["plan_hash"],
                    "jobs": len(plan["jobs"]), "execution_manifest_hash": manifest["manifest_hash"], "time": time.time()})
                generate(cp)
        from .c_repair_pipeline import finalize
        finalize(Path(manifest["canonical_pipeline"]), read(manifest["canonical_pipeline"]))
        write_json(control / "complete.json", {"all_pass": True, "manifest_hash": manifest["manifest_hash"],
            "runtime_strata_must_remain_distinguished": True, "time": time.time()})


def launch(path):
    manifest = load(path)
    require_cpu_validation(path.parent, manifest)
    session = manifest["session"]
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError("Isolated execution session already exists")
    plan = load_plan(Path(manifest["campaigns"][0]))
    cmd = ["env", "PYTHONPATH=" + str(Path(plan["project"]) / "repo"), plan["python"], "-u", "-m",
           "medtoolbench.supplement.isolated_execution", "--manifest", str(path)]
    log = path.parent / "controller.log"
    shell = "exec " + shlex.join(cmd) + " >> " + shlex.quote(str(log)) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "supervisor", "-c", plan["project"], shell], check=True)
    write_json(path.parent / "launch.json", {"manifest_hash": manifest["manifest_hash"], "session": session, "time": time.time()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--launch", action="store_true")
    a = p.parse_args()
    (launch if a.launch else run)(a.manifest.resolve())
