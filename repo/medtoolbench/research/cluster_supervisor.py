"""Cross-node scheduling of the original frozen jobs, with shared-file leases.

Start only after the legacy two-worker controller has been stopped and this
handoff has been recorded. Whole-batch recovery and independent auditing are
the original frozen implementations. No accuracy-based retries or exclusions.
"""
import argparse
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

from .audit import audit
from .cluster_execution import FileLease, deployment, validate_worker
from .job_state import write_json
from .supervisor import command, ready_completion, stop_owned


def attempt_dirs(folder):
    return sorted((p for p in folder.glob("attempt*") if p.is_dir() and p.name[7:].isdigit()),
                  key=lambda p: int(p.name[7:]))


def next_attempt(folder, maximum):
    previous = attempt_dirs(folder)
    if not previous:
        return folder / "attempt1", None
    last = previous[-1]
    if (last / "complete.json").exists():
        raise RuntimeError("Completed attempt must be audited before retry selection")
    if not (last / "infrastructure_failure.txt").exists():
        raise RuntimeError(f"Unresolved prior attempt {last}; never assume a missing heartbeat means dead")
    number = int(last.name[7:]) + 1
    if number > maximum:
        raise RuntimeError("Frozen infrastructure attempt budget exhausted")
    return folder / f"attempt{number}", last if (last / "config.json").exists() else None


def run_command(plan, job, system, output, resume, config_path, worker):
    cmd = command(plan, job, system, output, resume)
    index = cmd.index("medtoolbench.research.run")
    cmd[index:index + 1] = ["medtoolbench.research.cluster_execution", "--deployment", str(config_path),
                          "--worker", worker]
    return cmd


def execute(cmd, log, status, row, inherited_fds, stall_seconds=7200, wall_seconds=64800):
    output = Path(cmd[cmd.index("--out") + 1])
    started = last_progress = time.monotonic()
    signature = None
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(row["gpu"]))
    with log.open("x") as stream:
        proc = subprocess.Popen(cmd, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True, pass_fds=tuple(inherited_fds))
        try:
            while proc.poll() is None:
                progress = output / "progress.json"
                if progress.exists():
                    current = (progress.stat().st_mtime_ns, progress.stat().st_size)
                    if current != signature:
                        signature, last_progress = current, time.monotonic()
                elapsed = time.monotonic() - started
                reason = ("wall-time limit" if elapsed > wall_seconds else
                          "no-progress timeout" if time.monotonic() - last_progress > stall_seconds else None)
                write_json(status, {"state": "running", "worker": row, "pid": proc.pid,
                                    "controller_pid": os.getpid(), "heartbeat": time.time(),
                                    "seconds": elapsed, "out": str(output)})
                if reason:
                    stop_owned(proc)
                    return {"returncode": proc.returncode, "timeout": reason}
                time.sleep(5)
            return {"returncode": proc.returncode, "timeout": None}
        finally:
            stop_owned(proc)


def finish_job(folder, plan, job, output, attempts, proof):
    if not proof["all_pass"] or proof["episodes"] != job["expected_episodes"]:
        raise RuntimeError("Completed run did not pass the frozen independent audit/count")
    result = {"state": "complete", "job": job["id"], "model": job["model"],
              "episodes": job["expected_episodes"], "run": str(output), "attempts": attempts}
    write_json(folder / "result.json", result)


def process_job(root, config_path, config, plan, job, row, worker_lease, job_lease):
    folder = root / "jobs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    history = folder / "attempts.json"
    attempts = json.loads(history.read_text()) if history.exists() else []
    existing = attempt_dirs(folder)
    if existing and (existing[-1] / "complete.json").exists():
        output = existing[-1]
        proof = audit(output, Path(plan["benchmark"]))
        if not proof["all_pass"]:
            raise RuntimeError("Previously completed attempt failed audit; requires explicit investigation")
        attempts.append({"attempt": int(output.name[7:]), "returncode": 0, "audit": proof,
                         "recovered_after_controller_handoff": True})
        finish_job(folder, plan, job, output, attempts, proof)
        return
    if existing and not any(r.get("attempt") == int(existing[-1].name[7:]) for r in attempts):
        attempts.append({"attempt": int(existing[-1].name[7:]), "returncode": 1,
                         "recovered_after_controller_handoff": True, "audit": None})
    systems = {s["name"]: s for s in plan["systems"]}
    while True:
        # Check core and resource extension at every attempt, not only startup.
        deployment(config_path)
        output, resume = next_attempt(folder, plan["failure_policy"]["max_attempts"])
        cmd = run_command(plan, job, systems[job["model"]], output, resume, config_path, row["name"])
        write_json(folder / f"command{output.name[7:]}.json", {"command": cmd, "worker": row,
                   "deployment_hash": config["deployment_hash"]})
        execution = execute(cmd, folder / (output.name + ".cluster.log"),
                            config_path.parent / (row["name"] + ".json"), row,
                            [worker_lease.fileno(), job_lease.fileno()])
        proof = audit(output, Path(plan["benchmark"])) if (output / "complete.json").exists() else None
        attempts.append({"attempt": int(output.name[7:]), **execution, "audit": proof,
                         "worker": row["name"]})
        write_json(history, attempts)
        if execution["returncode"] == 0 and proof and proof["all_pass"]:
            finish_job(folder, plan, job, output, attempts, proof)
            return
        # No retries based on task accuracy. Audit-invalid completed evidence is
        # retained and fails closed rather than being silently cherry-picked.
        if proof or int(output.name[7:]) >= plan["failure_policy"]["max_attempts"]:
            raise RuntimeError("Infrastructure/audit attempt limit reached")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deployment", type=Path, required=True)
    p.add_argument("--worker", required=True)
    a = p.parse_args()
    config_path = a.deployment.resolve()
    config, plan = deployment(config_path)
    row = validate_worker(config, a.worker)
    root = Path(config["study"])
    control = config_path.parent
    handoff = json.loads((control / "handoff.json").read_text())
    if not handoff.get("legacy_controllers_stopped") or handoff["plan_hash"] != plan["plan_hash"]:
        raise RuntimeError("Legacy controller handoff is not complete")
    status = control / (a.worker + ".json")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Cluster worker received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    with FileLease(control / "locks" / (a.worker + ".lock")) as worker_lease:
        try:
            while True:
                deployment(config_path)
                picked = False
                for job in plan["jobs"]:
                    folder = root / "jobs" / job["id"]
                    if (folder / "result.json").exists():
                        continue
                    lease = FileLease(control / "locks" / (job["id"] + ".lock"))
                    try:
                        lease.__enter__()
                    except BlockingIOError:
                        continue
                    try:
                        if (folder / "result.json").exists():
                            continue
                        picked = True
                        write_json(control / (a.worker + "-assignment.json"),
                                   {"job": job["id"], "worker": row, "heartbeat": time.time()})
                        try:
                            process_job(root, config_path, config, plan, job, row, worker_lease, lease)
                        except Exception:
                            # Keep independent jobs moving; never hide a failed shard.
                            write_json(folder / "result.json", {"state": "failed", "job": job["id"],
                                       "model": job["model"], "episodes": 0, "worker": row,
                                       "error": traceback.format_exc()})
                    finally:
                        lease.__exit__()
                    break
                if picked:
                    continue
                final = ready_completion(root, plan)
                if final:
                    try:
                        with FileLease(control / "locks" / "final-report.lock"):
                            if not (root / "complete.json").exists():
                                write_json(status, {"state": "reporting", "worker": row, "heartbeat": time.time()})
                                from .study_report import generate
                                final["report"] = generate(root, plan)
                                final["execution_deployment"] = str(config_path)
                                if final["episodes"] != plan["expected_episodes"]:
                                    raise RuntimeError("Final episode count mismatch")
                                write_json(root / "complete.json", final)
                    except BlockingIOError:
                        pass
                    write_json(status, {"state": "complete", "worker": row, "heartbeat": time.time()})
                    return
                unfinished = [j for j in plan["jobs"] if not (root / "jobs" / j["id"] / "result.json").exists()]
                if not unfinished:
                    write_json(status, {"state": "failed", "worker": row, "heartbeat": time.time(),
                                        "error": "All jobs terminal but at least one failed"})
                    raise SystemExit(2)
                write_json(status, {"state": "waiting_for_other_workers", "worker": row, "heartbeat": time.time()})
                time.sleep(15)
        except BaseException:
            write_json(status, {"state": "interrupted_or_failed", "worker": row,
                                "heartbeat": time.time(), "error": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
