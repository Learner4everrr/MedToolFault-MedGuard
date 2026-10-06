"""Six-GPU shared-lock queue, development gates, bounded infrastructure retries."""
import argparse
import json
import os
import shlex
import subprocess
import time
import traceback
from pathlib import Path

from medtoolbench.research.cluster_execution import FileLease, short_host, validate_worker
from medtoolbench.research.cluster_supervisor import execute
from medtoolbench.research.job_state import write_json
from .runner import load_plan


def gate(root, plan):
    if plan.get("require_oracle"):
        oracle = root / "oracle" / "complete.json"
        if (root / "oracle" / "failed.json").exists():
            return "failed"
        if not oracle.exists():
            return "waiting"
        proof = json.loads(oracle.read_text())
        if not proof["all_pass"] or proof["plan_hash"] != plan["plan_hash"]:
            return "failed"
    jobs = [j for j in plan["jobs"] if j["stage"] == "qualification"]
    results = [root / "jobs" / j["id"] / "result.json" for j in jobs]
    if not all(p.exists() for p in results):
        return "waiting"
    return "passed" if all(json.loads(p.read_text())["state"] == "complete" for p in results) else "failed"


def worker_main(path, name):
    plan = load_plan(path)
    row = validate_worker(plan, name)
    root = path.parent
    status = root / "workers" / (name + ".json")
    status.parent.mkdir(exist_ok=True)
    with FileLease(root / "locks" / (name + ".lock")) as worker_lease:
        while True:
            plan = load_plan(path)
            state = gate(root, plan)
            if state == "failed":
                write_json(status, {"state": "qualification_failed", "worker": row, "heartbeat": time.time()})
                return
            picked = False
            for job in plan["jobs"]:
                if job["stage"] != "qualification" and state != "passed":
                    continue
                folder = root / "jobs" / job["id"]
                if (folder / "result.json").exists():
                    continue
                lease = FileLease(root / "locks" / (job["id"] + ".lock"))
                try:
                    lease.__enter__()
                except BlockingIOError:
                    continue
                try:
                    if (folder / "result.json").exists():
                        continue
                    picked = True
                    folder.mkdir(parents=True, exist_ok=True)
                    attempts = []
                    try:
                        if list(folder.glob("attempt*")):
                            raise RuntimeError("Unresolved earlier attempt; inspect before restarting")
                        for number in range(1, 3):
                            out = folder / ("attempt" + str(number))
                            module = plan.get("runner_module", "medtoolbench.supplement.runner")
                            cmd = [plan["python"], "-u", "-m", module, "--plan", str(path), "--worker", name,
                                   "--job", job["id"], "--out", str(out)]
                            write_json(folder / ("command" + str(number) + ".json"), {"command": cmd, "worker": row})
                            result = execute(cmd, folder / ("attempt" + str(number) + ".log"), status,
                                             row, [worker_lease.fileno(), lease.fileno()], stall_seconds=3600, wall_seconds=64800)
                            attempts.append(result)
                            if (out / "complete.json").exists() and result["returncode"] == 0:
                                audit = json.loads((out / "audit.json").read_text())
                                if not audit["all_pass"] or audit["episodes"] != len(job["episodes"]):
                                    raise RuntimeError("Completed evidence failed audit; never rerun based on outcomes")
                                write_json(folder / "result.json", {"state": "complete", "job": job["id"], "model": job["model"],
                                    "episodes": audit["episodes"], "run": str(out), "attempts": attempts, "worker": name})
                                break
                            if (out / "audit.json").exists() or number == 2:
                                raise RuntimeError("Audit failure or infrastructure retry budget exhausted")
                    except Exception:
                        write_json(folder / "result.json", {"state": "failed", "job": job["id"], "error": traceback.format_exc(), "attempts": attempts})
                finally:
                    lease.__exit__()
                break
            if picked:
                continue
            results = [root / "jobs" / j["id"] / "result.json" for j in plan["jobs"]]
            if all(p.exists() for p in results):
                rows = [json.loads(p.read_text()) for p in results]
                good = all(r["state"] == "complete" for r in rows)
                try:
                    with FileLease(root / "locks" / "finish.lock"):
                        write_json(root / ("complete.json" if good else "failed.json"), {"all_pass": good, "jobs": len(rows),
                            "episodes": sum(r.get("episodes", 0) for r in rows), "plan_hash": plan["plan_hash"], "finished_at": time.time(),
                            "analysis_status": "Per-run audits complete; aggregate statistical report is separate"})
                except BlockingIOError:
                    pass
                write_json(status, {"state": "complete" if good else "failed", "worker": row, "heartbeat": time.time()})
                return
            write_json(status, {"state": "waiting", "qualification": state, "worker": row, "heartbeat": time.time()})
            time.sleep(10)


def launch(path):
    plan = load_plan(path)
    rows = [r for r in plan["workers"] if r["host"] == short_host()]
    for row in rows:
        validate_worker(plan, row["name"])
    session = plan["session"]
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError("Supplement tmux session already exists; inspect it")
    root = Path(plan["project"])
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "control", "-c", str(root)], check=True)
    subprocess.run(["tmux", "set-option", "-t", session, "remain-on-exit", "on"], check=True)
    for row in rows:
        cmd = ["env", "PYTHONPATH=" + str(root / "repo"), plan["python"], "-u", "-m", "medtoolbench.supplement.campaign",
               "--plan", str(path), "--worker", row["name"]]
        subprocess.run(["tmux", "new-window", "-t", session, "-n", "gpu" + str(row["gpu"]), "-c", str(root), "exec " + shlex.join(cmd)], check=True)
    write_json(path.parent / (short_host() + "-launch.json"), {"workers": rows, "session": session, "time": time.time()})
    print(json.dumps({"host": short_host(), "workers_launched": len(rows), "session": session}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--worker")
    p.add_argument("--launch", action="store_true")
    a = p.parse_args()
    if a.launch:
        launch(a.plan.resolve())
    elif a.worker:
        worker_main(a.plan.resolve(), a.worker)
    else:
        raise SystemExit("Choose --launch or --worker")


if __name__ == "__main__":
    main()
