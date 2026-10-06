"""Explicit quota-stop reconciliation and continuation of unchanged inference.

Original execution sources, proofs, attempts and scientific plans are immutable.
Only process supervision/checkpoint persistence changes. Partial shards restart
as whole shards; no outcome-based filtering or row pooling is performed.
"""
import argparse
import errno
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.build import load_gzip
from medtoolbench.research.cluster_execution import FileLease, validate_worker
from medtoolbench.research.supervisor import stop_owned
from .isolated_manifest import load as load_execution, read, require_cpu_validation, preflight_passes, proof_passes
from .isolated_execution import IntegrityError, model_gate, failure_kind, validate_retry_history
from .memory_execution import verify_completion
from .runner import load_plan

STORAGE_ERRORS = {errno.EDQUOT, errno.ENOSPC}


def atomic_json(path, value):
    """Unique temp name, fsync before publish; clean only our own failed temp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", prefix=path.name + ".", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            name = stream.name
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def write_probe(root):
    """Exercise create, fsync, rename, readback, cleanup on the actual mount."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="write-probe-", dir=root) as folder:
        path = Path(folder) / "probe.json"
        value = {"bytes": "x" * 1048576, "time": time.time()}
        atomic_json(path, value)
        if read(path) != value:
            raise IOError("Storage readback differs from written content")
    return {"all_pass": True, "bytes": 1048576, "fsync_and_readback": True, "time": time.time()}


class Journal:
    """Local emergency journal; remote heartbeat failure alone never kills LLM."""
    def __init__(self, root, local):
        self.root, self.local = Path(root), Path(local)
        self.blocked = threading.Event()

    def put(self, path, value):
        path = Path(path)
        record = {"destination": str(path), "value": value, "time": time.time()}
        local_path = self.local / (digest(str(path)) + ".json")
        atomic_json(local_path, record)
        try:
            atomic_json(path, value)
        except OSError as error:
            if error.errno not in STORAGE_ERRORS:
                raise
            self.blocked.set()
            atomic_json(self.local / "storage_alert.json", {"errno": error.errno,
                "destination": str(path), "time": time.time(), "new_dispatch_paused": True})
            return False
        return True

    def persist(self, path, value):
        while not self.put(path, value):
            time.sleep(15)

    def healthy(self):
        try:
            write_probe(self.root / "storage_probes")
            self.blocked.clear()
            return True
        except OSError as error:
            if error.errno not in STORAGE_ERRORS:
                raise
            self.blocked.set()
            atomic_json(self.local / "storage_alert.json", {"errno": error.errno,
                "time": time.time(), "new_dispatch_paused": True})
            return False


def load(path):
    manifest = read(path)
    expected = manifest.pop("manifest_hash")
    if digest(manifest) != expected:
        raise IntegrityError("Storage recovery manifest changed")
    manifest["manifest_hash"] = expected
    for file, checksum in manifest["source_hashes"].items():
        if digest(Path(file).read_text()) != checksum:
            raise IntegrityError("Recovery source changed: " + file)
    base = load_execution(Path(manifest["execution_manifest"]))
    if base["manifest_hash"] != manifest["execution_manifest_hash"]:
        raise IntegrityError("Inference execution changed")
    require_cpu_validation(Path(manifest["execution_manifest"]).parent, base)
    for file, checksum in manifest["gate_hashes"].items():
        if digest(read(file)) != checksum:
            raise IntegrityError("Previously passed validation changed")
    return manifest, base


def inventory(base, control):
    completed, interrupted, historical = [], [], []
    old_control = Path(base["_path"]).parent
    state = read(old_control / "status.json")
    failure = read(old_control / "needs_review.json")
    if state["state"] != "integrity_stop" or "Disk quota exceeded" not in failure.get("integrity_error", ""):
        raise IntegrityError("Previous stop is not the diagnosed quota incident")
    worker_records = [read(p) for p in (old_control / "workers").glob("*.json")]
    owners = {r.get("out"): r for r in worker_records}
    for cp in base["campaigns"]:
        cp = Path(cp)
        plan = load_plan(cp)
        for job in plan["jobs"]:
            folder = cp.parent / "jobs" / job["id"]
            result = read(folder / "result.json")
            if result and result.get("state") == "complete":
                proof = verify_completion(Path(result["run"]), plan, job)
                completed.append({"campaign": str(cp), "job": job["id"],
                    "result_hash": digest(result), "audit_hash": digest(proof),
                    "episodes": len(job["episodes"]), "stage": job["stage"]})
                continue
            attempts = sorted((p for p in folder.glob("attempt*") if p.is_dir() and p.name[7:].isdigit()),
                              key=lambda p: int(p.name[7:]))
            if not attempts:
                if result:
                    raise IntegrityError("Terminal job has no attempt evidence")
                continue
            if result:
                history = validate_retry_history(folder)
                historical.append({"campaign": str(cp), "job": job["id"], **history})
                continue
            if len(attempts) != 1:
                raise IntegrityError("Unresolved multi-attempt history requires separate review")
            out = attempts[-1]
            config = read(out / "config.json")
            owner = owners.get(str(out))
            if not owner or owner["state"] != "running" or not config:
                raise IntegrityError("Interrupted attempt has no matching prior live-worker record")
            if config["plan_hash"] != plan["plan_hash"] or config["job"] != job or config.get("isolated_execution", {}).get("manifest_hash") != base["manifest_hash"]:
                raise IntegrityError("Interrupted attempt identity changed")
            if any((out / n).exists() for n in ("audit.json", "complete.json", "failure.json")):
                raise IntegrityError("Interrupted attempt needs a different reconciliation path")
            if abs(owner["heartbeat"] - state["heartbeat"]) > 600:
                raise IntegrityError("Attempt was not active near the quota-stop window")
            wanted = {e["key"] for e in job["episodes"]}
            saved, files = set(), {}
            for file in sorted((out / "episodes").glob("*.json.gz")):
                row = load_gzip(file)
                if row["key"] not in wanted or row["key"] in saved:
                    raise IntegrityError("Duplicate or unexpected saved partial row")
                saved.add(row["key"])
                files[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            interrupted.append({"campaign": str(cp), "job": job["id"], "out": str(out),
                "owner": owner, "config_hash": digest(config), "saved_episodes": len(saved),
                "expected_episodes": len(wanted), "partial_file_hashes": files,
                "partial_rows_pooled": False, "reason": "Confirmed previous controller quota-stop window; dead ownership and exclusive locks checked at handoff"})
    record = {"completed": completed, "interrupted": interrupted, "historical_failures": historical,
              "old_stop_hash": digest(state), "old_failure_hash": digest(failure), "time": time.time()}
    atomic_json(control / "reconciliation.json", record)
    return record


def prepare(args):
    base = load_execution(args.execution_manifest.resolve())
    require_cpu_validation(args.execution_manifest.parent, base)
    if not preflight_passes(args.execution_manifest.parent, base) or not proof_passes(args.execution_manifest.parent / "gemma_development/complete.json", base):
        raise IntegrityError("All previous Gemma gates must remain passed")
    if args.out.exists():
        raise IntegrityError("Never overwrite a recovery revision")
    args.out.mkdir(parents=True)
    atomic_json(args.out / "write_probe.json", write_probe(args.out / "storage_probes"))
    base["_path"] = str(args.execution_manifest.resolve())
    snapshot = inventory(base, args.out.resolve())
    root = Path(__file__).resolve().parents[3]
    files = [Path(__file__).resolve(), root / "repo/tests/test_storage_recovery.py"]
    gates = [args.execution_manifest.parent / name for name in (
        "validation/unit.json", "validation/hapi.json", "preflight/compare/complete.json",
        "preflight/stress/complete.json", "gemma_development/complete.json")]
    manifest = {"created_at": time.time(), "execution_manifest": str(args.execution_manifest.resolve()),
        "execution_manifest_hash": base["manifest_hash"], "canonical_pipeline": base["canonical_pipeline"],
        "source_hashes": {str(f): digest(f.read_text()) for f in files},
        "gate_hashes": {str(f.resolve()): digest(read(f)) for f in gates},
        "reconciliation_hash": digest(snapshot), "session": "medtoolbench-storage-recovery-v1",
        "max_attempts_this_recovery": 2,
        "retry_policy": "Only a new quota/ENOSPC failure can use a second attempt after three successful write probes. OOM and other model failures are quarantined without blind retries. All completed original jobs retained; interrupted partial shards restart whole in new directories.",
        "scheduling_policy": "Reconcile and prioritize the eight interrupted assignments, then round-robin eligible campaigns, with original qualification/oracle gates. Prior known failed Gemma shards are eligible under already validated serial execution.",
        "persistence_policy": "Local emergency journal survives shared quota failure; pause new dispatch on failed write probes. Remote heartbeat failure alone never terminates a healthy model. Child output failure still remains infrastructure failure, never a scored clinical outcome.",
        "inference_configuration_unchanged": True, "inference_api": False}
    manifest["manifest_hash"] = digest(manifest)
    atomic_json(args.out / "manifest.json", manifest)
    for file in files:
        target = args.out / "frozen_source" / file.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    print(json.dumps({"manifest_hash": manifest["manifest_hash"], "completed_jobs_retained": len(snapshot["completed"]),
        "quota_interrupted_jobs": len(snapshot["interrupted"]), "partial_rows_preserved_not_pooled": sum(r["saved_episodes"] for r in snapshot["interrupted"]),
        "historical_failed_jobs": len(snapshot["historical_failures"])}), flush=True)


def execute(cmd, log, worker, control, fds, journal):
    out = Path(cmd[cmd.index("--out") + 1])
    start = last = time.monotonic()
    signature = None
    with log.open("x") as stream:
        proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker["gpu"])),
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(fds))
        try:
            while proc.poll() is None:
                progress = out / "progress.json"
                if progress.exists():
                    current = (progress.stat().st_mtime_ns, progress.stat().st_size)
                    if current != signature:
                        signature, last = current, time.monotonic()
                elapsed = time.monotonic() - start
                journal.put(control / "workers" / (worker["name"] + ".json"), {"state": "running", "worker": worker,
                    "pid": proc.pid, "controller_pid": os.getpid(), "heartbeat": time.time(), "out": str(out), "seconds": elapsed})
                timeout = "wall-time limit" if elapsed > 64800 else "no-progress timeout" if time.monotonic() - last > 3600 else None
                if timeout:
                    stop_owned(proc)
                    return {"returncode": proc.returncode, "timeout": timeout}
                time.sleep(5)
            return {"returncode": proc.returncode, "timeout": None}
        finally:
            stop_owned(proc)


def attempt_job(path, cp, job, worker, fds, journal):
    manifest, base = load(path)
    control, plan = path.parent, load_plan(cp)
    folder = cp.parent / "jobs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = read(control / "reconciliation.json")
    with FileLease(cp.parent / "locks" / (job["id"] + ".lock")) as lease:
        existing = read(folder / "result.json")
        if existing and existing["state"] == "complete":
            raise IntegrityError("Completed job cannot be restarted")
        marker = folder / "storage_recovery_v1_authorization.json"
        if marker.exists():
            raise IntegrityError("Recovery assignment already attempted")
        old = sorted((p for p in folder.glob("attempt*") if p.is_dir() and p.name[7:].isdigit()), key=lambda p: int(p.name[7:]))
        approved = next((r for r in snapshot["interrupted"] if r["campaign"] == str(cp) and r["job"] == job["id"]), None)
        if old:
            if approved:
                if old[-1] != Path(approved["out"]) or digest(read(old[-1] / "config.json")) != approved["config_hash"]:
                    raise IntegrityError("Interrupted attempt changed after reconciliation")
            else:
                validate_retry_history(folder)
        authorization = {"manifest_hash": manifest["manifest_hash"], "job": job["id"],
            "prior_attempts": [str(p) for p in old], "prior_result": existing,
            "quota_interruption": approved, "partial_rows_pooled": False, "time": time.time()}
        journal.persist(marker, authorization)
        number = max((int(p.name[7:]) for p in old), default=0) + 1
        records = []
        for offset in range(manifest["max_attempts_this_recovery"]):
            out = folder / ("attempt" + str(number + offset))
            log = folder / (out.name + ".storage.log")
            cmd = [plan["python"], "-u", "-m", "medtoolbench.supplement.isolated_runner", "--execution-manifest",
                manifest["execution_manifest"], "--plan", str(cp), "--job", job["id"], "--worker", worker["name"], "--out", str(out)]
            journal.persist(folder / ("command" + str(number + offset) + ".json"), {"command": cmd, "worker": worker,
                "execution_manifest_hash": base["manifest_hash"], "storage_recovery_manifest_hash": manifest["manifest_hash"]})
            launch_error = ""
            try:
                run = execute(cmd, log, worker, control, [*fds, lease.fileno()], journal)
            except OSError as error:
                if error.errno not in STORAGE_ERRORS:
                    raise
                journal.blocked.set()
                launch_error = str(error)
                run = {"returncode": None, "timeout": None, "storage_error": launch_error}
            records.append({"attempt": number + offset, **run})
            result = {"job": job["id"], "model": job["model"], "worker": worker["name"], "run": str(out),
                "attempts": records, "execution_manifest": manifest["execution_manifest"],
                "execution_manifest_hash": base["manifest_hash"], "storage_recovery_manifest": str(path),
                "storage_recovery_manifest_hash": manifest["manifest_hash"]}
            if run["returncode"] == 0 and (out / "complete.json").exists():
                proof = verify_completion(out, plan, job)
                result.update(state="complete", episodes=proof["episodes"])
            else:
                if (out / "audit.json").exists() or (out / "complete.json").exists():
                    raise IntegrityError("Audit-invalid evidence cannot be retried")
                error = launch_error or (read(out / "failure.json") or {}).get("error", "")
                tail = log.read_text(errors="replace")[-65536:] if log.exists() else ""
                quota = any(s in error + tail for s in ("Disk quota exceeded", "No space left on device"))
                if quota and offset + 1 < manifest["max_attempts_this_recovery"]:
                    journal.blocked.set()
                    journal.persist(control / "quota_incidents" / (job["id"] + "-" + out.name + ".json"),
                        {"run": str(out), "worker": worker, "error": error or tail[-4000:], "time": time.time()})
                    consecutive = 0
                    while consecutive < 3:
                        time.sleep(15)
                        consecutive = consecutive + 1 if journal.healthy() else 0
                    continue
                result.update(state="quarantined", episodes=0, assigned_episodes=len(job["episodes"]),
                    failure_kind="storage_quota" if quota else failure_kind(error or tail), error=error or tail[-4000:])
            journal.persist(folder / "result.json", result)
            journal.put(control / "workers" / (worker["name"] + ".json"), {"state": "idle", "worker": worker,
                "finished_job": job["id"], "result": result["state"], "heartbeat": time.time()})
            return result


def candidates(plans, snapshot, active, blocked):
    interrupted = {(r["campaign"], r["job"]) for r in snapshot["interrupted"]}
    eligible = []
    for index, (cp, plan) in enumerate(plans):
        gates = {}
        for job in plan["jobs"]:
            key = str(cp), job["id"]
            if key in active or job["model"] in blocked:
                continue
            folder = cp.parent / "jobs" / job["id"]
            result = read(folder / "result.json")
            if result and result["state"] == "complete" or (folder / "storage_recovery_v1_authorization.json").exists():
                continue
            if job["model"] not in gates:
                gates[job["model"]] = model_gate(cp, plan, job["model"])
            if job["stage"] == "qualification" or gates[job["model"]] == "passed":
                eligible.append((key not in interrupted, index, cp, plan, job))
    return eligible


def assert_no_old_processes(base, plans):
    from medtoolbench.supplement.isolated_manifest import read
    old = Path(base["_path"]).parent
    rows = [read(old / "started.json")] + [read(p) for p in (old / "workers").glob("*.json")]
    for row in rows:
        if row and row.get("pid") and Path("/proc" + "/" + str(row["pid"])).exists():
            raise IntegrityError("Previous ownership PID still exists; inspect before restart")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    if apps.strip():
        raise IntegrityError("GPU compute processes still running")
    for worker in plans[0][1]["workers"]:
        validate_worker(plans[0][1], worker["name"])


def run(path):
    manifest, base = load(path)
    control = path.parent
    proof = read(control / "unit_tests.json")
    if not proof or not proof["all_pass"] or proof["manifest_hash"] != manifest["manifest_hash"]:
        raise IntegrityError("Require recovery regression tests")
    snapshot = read(control / "reconciliation.json")
    if digest(snapshot) != manifest["reconciliation_hash"]:
        raise IntegrityError("Reconciliation changed")
    if (control / "started.json").exists():
        raise IntegrityError("Do not blindly relaunch a prior recovery controller")
    base["_path"] = manifest["execution_manifest"]
    plans = [(Path(cp), load_plan(Path(cp))) for cp in base["campaigns"]]
    assert_no_old_processes(base, plans)
    rows = plans[0][1]["workers"]
    local = Path(tempfile.mkdtemp(prefix="medtoolbench-storage-recovery-"))
    journal = Journal(control, local)
    write_probe(control / "storage_probes")
    (control / "workers").mkdir(exist_ok=True)
    (control / "quota_incidents").mkdir(exist_ok=True)
    with ExitStack() as stack:
        leases = [stack.enter_context(FileLease(control / "supervisor.lock")),
                  stack.enter_context(FileLease(Path(manifest["execution_manifest"]).parent / "supervisor.lock"))]
        for root in base["legacy_worker_lock_roots"] + [str(cp.parent) for cp, _ in plans]:
            for row in rows:
                leases.append(stack.enter_context(FileLease(Path(root) / "locks" / (row["name"] + ".lock"))))
        # Verify no orphan has inherited a pending-job lease before reuse.
        for cp, plan in plans:
            for job in plan["jobs"]:
                with FileLease(cp.parent / "locks" / (job["id"] + ".lock")):
                    pass
        for row in snapshot["completed"]:
            result = read(Path(row["campaign"]).parent / "jobs" / row["job"] / "result.json")
            if digest(result) != row["result_hash"]:
                raise IntegrityError("Completed result changed after reconciliation")
        fds = [lease.fileno() for lease in leases]
        journal.persist(control / "started.json", {"pid": os.getpid(), "time": time.time(),
            "manifest_hash": manifest["manifest_hash"], "emergency_journal": str(local),
            "prior_owners_absent_and_locks_free": True})
        journal.persist(Path(manifest["execution_manifest"]).parent / "storage_successor.json",
            {"manifest": str(path), "manifest_hash": manifest["manifest_hash"], "time": time.time()})
        busy, blocked, fatal, cursor, last_probe = {}, {}, None, 0, 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            while True:
                try:
                    load(path)
                    if time.monotonic() - last_probe >= 60:
                        journal.healthy()
                        last_probe = time.monotonic()
                    for name, (future, key) in list(busy.items()):
                        if not future.done():
                            continue
                        try:
                            result = future.result()
                            if result["state"] != "complete":
                                blocked[result["model"]] = {"reason": result["failure_kind"], "job": result["job"], "run": result["run"]}
                        except Exception:
                            if fatal is None:
                                fatal = traceback.format_exc()
                        del busy[name]
                    active = {key for _, key in busy.values()}
                    if fatal is None and not journal.blocked.is_set():
                        for worker in rows:
                            if worker["name"] in busy:
                                continue
                            ready = candidates(plans, snapshot, active, blocked)
                            if not ready:
                                continue
                            ready.sort(key=lambda x: (x[0], (x[1] - cursor) % len(plans), x[4]["id"]))
                            _, index, cp, plan, job = ready[0]
                            cursor = (index + 1) % len(plans)
                            key = str(cp), job["id"]
                            busy[worker["name"]] = (pool.submit(attempt_job, path, cp, job, worker, fds, journal), key)
                            active.add(key)
                    pending, done = [], 0
                    for cp, plan in plans:
                        for job in plan["jobs"]:
                            result = read(cp.parent / "jobs" / job["id"] / "result.json")
                            if result and result["state"] == "complete":
                                done += len(job["episodes"])
                            else:
                                pending.append({"campaign": str(cp), "job": job["id"], "model": job["model"],
                                    "stage": job["stage"], "assigned_episodes": len(job["episodes"]),
                                    "state": result["state"] if result else "pending"})
                    journal.put(control / "pending.json", {"jobs": pending, "audited_episodes_including_development": done,
                        "pending_episodes": sum(r["assigned_episodes"] for r in pending), "time": time.time()})
                    state = "integrity_stop" if fatal else "storage_paused" if journal.blocked.is_set() else "running" if busy else "needs_review" if pending else "inference_complete"
                    journal.put(control / "status.json", {"state": state, "manifest_hash": manifest["manifest_hash"],
                        "heartbeat": time.time(), "pid": os.getpid(), "busy": {n: list(k) for n, (_, k) in busy.items()},
                        "blocked_models": blocked, "pending_jobs": len(pending), "emergency_journal": str(local)})
                    if fatal:
                        journal.put(control / "integrity_failure.json", {"error": fatal, "time": time.time()})
                    if not busy and not journal.blocked.is_set():
                        if fatal or pending:
                            journal.persist(control / "needs_review.json", {"error": fatal, "blocked_models": blocked, "pending_jobs": len(pending), "time": time.time()})
                            return
                        break
                except OSError as error:
                    if error.errno in STORAGE_ERRORS:
                        journal.blocked.set()
                        atomic_json(local / "storage_alert.json", {"error": str(error), "time": time.time()})
                    elif fatal is None:
                        fatal = traceback.format_exc()
                except Exception:
                    if fatal is None:
                        fatal = traceback.format_exc()
                time.sleep(5)
        from .report import generate
        for cp, plan in plans:
            with FileLease(cp.parent / "locks/statistical-report.lock"):
                journal.persist(cp.parent / "complete.json", {"all_pass": True, "plan_hash": plan["plan_hash"],
                    "jobs": len(plan["jobs"]), "storage_recovery_manifest_hash": manifest["manifest_hash"], "time": time.time()})
                generate(cp)
        from .c_repair_pipeline import finalize
        finalize(Path(base["canonical_pipeline"]), read(base["canonical_pipeline"]))
        journal.persist(control / "complete.json", {"all_pass": True, "manifest_hash": manifest["manifest_hash"], "time": time.time()})


def launch(path):
    manifest, base = load(path)
    session = manifest["session"]
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise IntegrityError("Recovery session already exists")
    plan = load_plan(Path(base["campaigns"][0]))
    cmd = ["env", "PYTHONPATH=" + str(Path(plan["project"]) / "repo"), plan["python"], "-u", "-m",
           "medtoolbench.supplement.storage_recovery", "run", "--manifest", str(path)]
    shell = "exec " + shlex.join(cmd) + " >> " + shlex.quote(str(path.parent / "controller.log")) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "supervisor", "-c", plan["project"], shell], check=True)
    atomic_json(path.parent / "launch.json", {"session": session, "manifest_hash": manifest["manifest_hash"], "time": time.time()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prepare_p = sub.add_parser("prepare")
    prepare_p.add_argument("--execution-manifest", type=Path, default=Path("data/supplement_isolated_execution_v1/manifest.json"))
    prepare_p.add_argument("--out", type=Path, default=Path("data/supplement_storage_recovery_v1"))
    for name in ("run", "launch"):
        s = sub.add_parser(name)
        s.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        (run if args.command == "run" else launch)(args.manifest.resolve())
