"""Stop only verified idle project controllers, reservation owner LAST.

Never kills a model or touches the user's Codex tmux session. A partial handoff
is preserved and must be reviewed, not retried blindly.
"""
import argparse
import json
import os
import signal
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

from medtoolbench.research.cluster_execution import FileLease, validate_worker
from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.isolated_manifest import load, read, require_cpu_validation
from medtoolbench.supplement.runner import load_plan


def processes():
    rows = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            args = [s.decode() for s in (path / "cmdline").read_bytes().split(b"\0") if s]
            stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
            rows.append({"pid": int(path.name), "args": args, "ppid": int(stat[1]),
                "start_ticks": stat[19], "cgroup": (path / "cgroup").read_text()})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return rows


def matches(row, target):
    args = row["args"]
    return ("-m" in args and args[args.index("-m") + 1] == target["module"]
            and target["argument"] in args and args[args.index(target["argument"]) + 1] == target["path"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    manifest = load(a.manifest)
    control = a.manifest.parent
    require_cpu_validation(control, manifest)
    if (control / "handoff.before.json").exists() or (control / "handoff.json").exists():
        raise RuntimeError("Handoff already attempted; inspect before retry")
    plan = load_plan(Path(manifest["campaigns"][0]))
    for worker in plan["workers"]:
        validate_worker(plan, worker["name"])
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    if any(int(line.split(",")[1]) != 0 for line in inventory.strip().splitlines()):
        raise RuntimeError("GPU memory is in use; refuse idle-controller handoff")
    all_rows, groups = processes(), []
    for target in manifest["legacy_controllers"]:
        selected = [r for r in all_rows if matches(r, target)]
        if len(selected) != target["expected_processes"]:
            raise RuntimeError("Controller identity/count changed: " + target["module"])
        for row in selected:
            if f"/job_{plan['workers'][0]['slurm_job']}/" not in row["cgroup"]:
                raise RuntimeError("Controller outside expected allocation")
            if any(r["ppid"] == row["pid"] for r in all_rows):
                raise RuntimeError("Controller still has a live child; refuse interruption")
        groups.append(selected)
    memory_root = Path(manifest["legacy_controllers"][-1]["path"]).parent
    if read(memory_root / "status.json")["state"] != "failed_closed":
        raise RuntimeError("Old reservation owner no longer failed/idle")
    for worker in plan["workers"]:
        old = read(Path(manifest["campaigns"][0]).parent / "workers" / (worker["name"] + ".json"))
        if not old or old["state"] != "waiting":
            raise RuntimeError("Old inference worker not at an idle boundary")
    write_json(control / "handoff.before.json", {"controllers": groups, "gpu_inventory": inventory,
        "manifest_hash": manifest["manifest_hash"], "time": time.time()})
    stopped = []
    # First stop all old dispatch/continuation workers while the memory owner
    # STILL holds 742 pending-job locks. Only then release its reservations.
    for selected in (groups[0] + groups[1], groups[2]):
        for row in selected:
            fresh = next((r for r in processes() if r["pid"] == row["pid"]), None)
            if not fresh or fresh["args"] != row["args"] or fresh["start_ticks"] != row["start_ticks"]:
                raise RuntimeError("PID identity changed before termination")
            os.kill(row["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 45
        pids = {r["pid"] for r in selected}
        while pids.intersection(r["pid"] for r in processes()):
            if time.monotonic() > deadline:
                raise RuntimeError("Controller did not stop cleanly; no SIGKILL escalation")
            time.sleep(.5)
        stopped.extend(selected)
    # Prove compatibility locks are free, not merely that PIDs disappeared.
    with ExitStack() as stack:
        for root in manifest["legacy_worker_lock_roots"] + [str(Path(p).parent) for p in manifest["campaigns"]]:
            for worker in plan["workers"]:
                stack.enter_context(FileLease(Path(root) / "locks" / (worker["name"] + ".lock")))
        for cp in manifest["campaigns"]:
            source = Path(cp)
            for job in load_plan(source)["jobs"]:
                stack.enter_context(FileLease(source.parent / "locks" / (job["id"] + ".lock")))
    write_json(control / "handoff.json", {"all_legacy_controllers_stopped": True,
        "stopped": stopped, "active_model_processes_interrupted": False,
        "compatibility_worker_and_job_locks_verified_free": True, "reservation_owner_stopped_last": True,
        "user_tmux_session_touched": False, "manifest_hash": manifest["manifest_hash"], "time": time.time()})
    write_json(memory_root / "superseded.json", {"successor": str(a.manifest.resolve()),
        "reason": "Explicit fault-isolating controller handoff", "original_failure_preserved": True, "time": time.time()})
    print(json.dumps({"stopped_idle_controllers": len(stopped), "active_models_interrupted": 0}), flush=True)


if __name__ == "__main__":
    main()
