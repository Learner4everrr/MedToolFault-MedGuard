"""Recorded continuation into separately frozen campaigns and audited reports."""
import argparse
import json
import os
import shlex
import subprocess
import time
import traceback
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import FileLease, short_host, validate_worker
from medtoolbench.research.job_state import write_json
from .campaign import worker_main
from .runner import load_plan


def load(path):
    config = json.loads(path.read_text())
    checksum = config.pop("pipeline_hash")
    if digest(config) != checksum:
        raise RuntimeError("Pipeline manifest changed")
    config["pipeline_hash"] = checksum
    for name, value in config["source_hashes"].items():
        if digest(Path(name).read_text()) != value:
            raise RuntimeError("Pipeline/report implementation changed")
    for campaign, expected in zip(config["campaigns"], config["plan_hashes"]):
        if load_plan(Path(campaign))["plan_hash"] != expected:
            raise RuntimeError("Continuation campaign identity changed")
    return config


def wait_campaign(root, status):
    while True:
        if (root / "complete.json").exists():
            return True
        if (root / "failed.json").exists():
            return False
        workers = list((root / "workers").glob("*.json"))
        if len(workers) == 6 and all(json.loads(p.read_text()).get("state") == "qualification_failed" for p in workers):
            return False
        write_json(status, {"state": "waiting_for_campaign", "campaign": str(root), "heartbeat": time.time()})
        time.sleep(15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", type=Path, required=True)
    p.add_argument("--worker")
    p.add_argument("--launch", action="store_true")
    a = p.parse_args()
    path = a.pipeline.resolve()
    config = load(path)
    first = load_plan(Path(config["campaigns"][0]))
    if a.launch:
        session = "medtoolbench-supplement-followup-v1"
        rows = [r for r in first["workers"] if r["host"] == short_host()]
        for row in rows:
            validate_worker(first, row["name"])
        if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
            raise RuntimeError("Follow-up session already exists")
        root = Path(first["project"])
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "control", "-c", str(root)], check=True)
        subprocess.run(["tmux", "set-option", "-t", session, "remain-on-exit", "on"], check=True)
        for row in rows:
            cmd = ["env", "PYTHONPATH=" + str(root / "repo"), first["python"], "-u", "-m", "medtoolbench.supplement.pipeline",
                   "--pipeline", str(path), "--worker", row["name"]]
            subprocess.run(["tmux", "new-window", "-t", session, "-n", "gpu" + str(row["gpu"]), "-c", str(root), "exec " + shlex.join(cmd)], check=True)
        write_json(path.parent / (short_host() + "-launch.json"), {"workers": rows, "time": time.time(), "session": session})
        print(json.dumps({"host": short_host(), "continuation_workers": len(rows)}))
        return
    if not a.worker:
        raise RuntimeError("Choose --worker or --launch")
    row = validate_worker(first, a.worker)
    status = path.parent / (a.worker + ".json")
    results = []
    with FileLease(path.parent / "locks" / (a.worker + ".lock")):
        try:
            for index, campaign in enumerate(config["campaigns"]):
                load(path)
                cp = Path(campaign)
                if index:
                    write_json(status, {"state": "executing_campaign", "campaign": campaign, "heartbeat": time.time()})
                    worker_main(cp, a.worker)
                complete = wait_campaign(cp.parent, status)
                results.append({"campaign": campaign, "all_pass": complete})
                if complete:
                    try:
                        with FileLease(cp.parent / "locks" / "statistical-report.lock"):
                            if not (cp.parent / "report/complete.json").exists():
                                from .report import generate
                                write_json(status, {"state": "reporting", "campaign": campaign, "heartbeat": time.time()})
                                generate(cp)
                    except BlockingIOError:
                        pass
                # Independent later experiments continue after a scientific/
                # qualification failure, but the whole project stays incomplete.
            write_json(status, {"state": "complete" if all(r["all_pass"] for r in results) else "completed_with_failures",
                "worker": row, "campaigns": results, "heartbeat": time.time()})
        except BaseException:
            write_json(status, {"state": "failed", "error": traceback.format_exc(), "campaigns": results, "heartbeat": time.time()})
            raise


if __name__ == "__main__":
    main()
