"""Execute a frozen primary matrix with bounded retries and terminal job states.

Two workers partition the job list. One failure does not block independent jobs;
every failed shard remains visible, and no completion marker is issued while any
shard is missing. Source changes stop execution rather than mixing versions.
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from medtoolbench.storage import digest
from .audit import audit
from .job_state import write_json
from .runtime import core_fingerprint


def load_plan(root):
    plan = json.loads((root / "plan.json").read_text())
    declared = plan.pop("plan_hash")
    if digest(plan) != declared:
        raise RuntimeError("Frozen plan was modified")
    plan["plan_hash"] = declared
    return plan


def command(plan, job, system, out, resume=None):
    limits = plan["generation"]
    cmd = [sys.executable, "-u", "-m", "medtoolbench.research.run", "--benchmark", plan["benchmark"],
           "--validation", plan["validation"], "--image", plan["image"], "--model", system["path"],
           "--out", str(out), "--split", "test", "--task-shards", str(job["task_shards"]),
           "--task-shard", str(job["task_shard"]), "--batch-size", str(system["batch_size"])]
    for key, value in limits.items():
        cmd += ["--" + key.replace("_", "-"), str(value)]
    cmd += ["--conditions", *plan["conditions"]]
    if system["thinking"]:
        cmd.append("--thinking")
    if resume:
        cmd += ["--resume-from", str(resume)]
    return cmd


def stop_owned(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def execute(cmd, env, log, status, stall_seconds=7200, wall_seconds=64800):
    started, last_progress = time.monotonic(), time.monotonic()
    out = Path(cmd[cmd.index("--out") + 1])
    signature = None
    with log.open("w") as stream:
        proc = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                progress = out / "progress.json"
                if progress.exists():
                    current = (progress.stat().st_mtime_ns, progress.stat().st_size)
                    if current != signature:
                        signature, last_progress = current, time.monotonic()
                reason = "wall-time limit" if time.monotonic() - started > wall_seconds else "no-progress timeout" if time.monotonic() - last_progress > stall_seconds else None
                write_json(status, {"state": "running", "pid": proc.pid, "heartbeat": time.time(),
                                    "seconds": time.monotonic() - started, "out": str(out)})
                if reason:
                    stop_owned(proc)
                    return {"returncode": proc.returncode, "timeout": reason}
                time.sleep(5)
            return {"returncode": proc.returncode, "timeout": None}
        finally:
            stop_owned(proc)


def ready_completion(root, plan):
    rows = []
    for job in plan["jobs"]:
        path = root / "jobs" / job["id"] / "result.json"
        if not path.exists():
            return None
        row = json.loads(path.read_text())
        if row["state"] != "complete":
            return None
        rows.append(row)
    return {"state": "complete", "jobs": len(rows), "episodes": sum(r["episodes"] for r in rows),
            "expected_episodes": plan["expected_episodes"], "plan_hash": plan["plan_hash"], "finished_at": time.time(),
            "runs": [r["run"] for r in rows], "phase": "primary held-out only; other research arms tracked separately"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--worker", type=int, choices=[0, 1], required=True)
    a = p.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Supervisor received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    root = a.study.resolve()
    plan = load_plan(root)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.worker))
    status = root / f"worker{a.worker}.json"
    lockpath = Path("/tmp") / f"medtoolbench-supervisor-{digest(str(root))[:16]}-{a.worker}.lock"
    with lockpath.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        systems = {s["name"]: s for s in plan["systems"]}
        failed = []
        try:
            for job in plan["jobs"][a.worker::2]:
                if core_fingerprint() != plan["core_hashes"]:
                    raise RuntimeError("Core source changed after freeze; do not continue this study version")
                folder = root / "jobs" / job["id"]
                folder.mkdir(parents=True, exist_ok=True)
                result_path = folder / "result.json"
                if result_path.exists():
                    existing = json.loads(result_path.read_text())
                    if existing["state"] == "complete":
                        continue
                    failed.append(job["id"])
                    continue
                attempts, resume = [], None
                for attempt in range(1, plan["failure_policy"]["max_attempts"] + 1):
                    out = folder / f"attempt{attempt}"
                    if out.exists():
                        # After a controller restart, explicitly recover only an
                        # already terminated attempt; never overwrite its evidence.
                        if (out / "complete.json").exists():
                            proof = audit(out, Path(plan["benchmark"]))
                            if proof["all_pass"]:
                                attempts.append({"attempt": attempt, "returncode": 0, "audit": proof})
                                break
                        if not (out / "infrastructure_failure.txt").exists():
                            raise RuntimeError(f"Unresolved prior attempt {out}; inspect live-process status before recovery")
                        resume = out
                        attempts.append({"attempt": attempt, "returncode": 1, "recovered_after_controller_restart": True})
                        continue
                    cmd = command(plan, job, systems[job["model"]], out, resume)
                    write_json(folder / f"command{attempt}.json", {"command": cmd, "gpu": a.worker})
                    execution = execute(cmd, env, folder / f"attempt{attempt}.log", status)
                    proof = None
                    if (out / "complete.json").exists():
                        proof = audit(out, Path(plan["benchmark"]))
                    entry = {"attempt": attempt, **execution, "audit": proof}
                    attempts.append(entry)
                    write_json(folder / "attempts.json", attempts)
                    if execution["returncode"] == 0 and proof and proof["all_pass"]:
                        break
                    # A complete but invalid audit gets a full fresh rerun, not
                    # checkpoint reuse. Low task accuracy never affects retrying.
                    resume = out if proof is None and (out / "config.json").exists() else None
                last_proof = attempts[-1].get("audit") if attempts else None
                success = bool(attempts) and attempts[-1].get("returncode") == 0 and bool(last_proof and last_proof.get("all_pass"))
                if success and last_proof["episodes"] != job["expected_episodes"]:
                    raise RuntimeError("Shard audit count differs from frozen job count")
                result = {"state": "complete" if success else "failed", "job": job["id"], "model": job["model"],
                          "episodes": job["expected_episodes"] if success else 0, "run": str(out), "attempts": attempts}
                write_json(result_path, result)
                if not success:
                    failed.append(job["id"])
                write_json(root / f"worker{a.worker}_progress.json", {"last_job": job["id"], "failed_jobs": failed, "heartbeat": time.time()})
            write_json(status, {"state": "failed" if failed else "complete", "failed_jobs": failed, "heartbeat": time.time()})
            if failed:
                write_json(root / f"worker{a.worker}_failed.json", {"failed_jobs": failed})
            final = ready_completion(root, plan)
            if final:
                if final["episodes"] != final["expected_episodes"]:
                    raise RuntimeError("Completed study has wrong episode count")
                from .study_report import generate
                final["report"] = generate(root, plan)
                write_json(root / "complete.json", final)
            if failed:
                raise SystemExit(2)
        except BaseException:
            write_json(status, {"state": "failed", "error": traceback.format_exc(), "heartbeat": time.time()})
            raise


if __name__ == "__main__":
    main()
