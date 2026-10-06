"""Append corrected C after existing workers finish, without touching active A/B/D."""
import argparse
import json
import shlex
import signal
import subprocess
import time
import traceback
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import FileLease, validate_worker
from medtoolbench.research.job_state import write_json
from .campaign import worker_main
from .pipeline import load as load_base
from .profile_plan import validate_repair
from .resource_pipeline import load as load_predecessor, wait_campaign
from .runner import load_plan


def load(path):
    config = load_base(path)
    previous, parents = load_predecessor(Path(config["predecessor_pipeline"]))
    if previous["pipeline_hash"] != config["predecessor_pipeline_hash"]:
        raise RuntimeError("Predecessor execution identity changed")
    if len(config["campaigns"]) != 4:
        raise RuntimeError("Expected unchanged A/B/D plus corrected C")
    for index in (0, 1, 3):
        if config["campaigns"][index] != previous["campaigns"][index] or config["plan_hashes"][index] != previous["plan_hashes"][index]:
            raise RuntimeError("Repair continuation must not replace A, B or D")
    repaired = load_plan(Path(config["campaigns"][2]))
    validate_repair(parents[2], repaired)
    if repaired["workers"] != parents[0]["workers"] or config["episodes"] != previous["episodes"]:
        raise RuntimeError("Repair changed resource assignment or episode matrix")
    if (path.parent / "paused.json").exists():
        raise RuntimeError("Repair continuation is explicitly paused")
    return config, repaired


def predecessor_ready(root, name):
    status = root / (name + ".json")
    if not status.exists():
        return False
    state = json.loads(status.read_text()).get("state")
    if state in {"failed", "interrupted_or_failed"}:
        raise RuntimeError("Predecessor worker failed; explicit recovery review required")
    return state in {"complete", "completed_with_failures"}


def launch(path):
    config, plan = load(path)
    proof = json.loads(Path(config["preflight"]).read_text())
    if not proof["all_pass"] or proof["plan_hash"] != plan["plan_hash"]:
        raise RuntimeError("Corrected real-HAPI regression must pass before continuation launch")
    session = config["session"]
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError("Repair session already exists")
    for row in plan["workers"]:
        validate_worker(plan, row["name"])
    root = Path(plan["project"])
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "control", "-c", str(root)], check=True)
    subprocess.run(["tmux", "set-option", "-t", session, "remain-on-exit", "on"], check=True)
    for row in plan["workers"]:
        cmd = ["env", "PYTHONPATH=" + str(root / "repo"), plan["python"], "-u", "-m",
            "medtoolbench.supplement.c_repair_pipeline", "--pipeline", str(path), "--worker", row["name"]]
        subprocess.run(["tmux", "new-window", "-t", session, "-n", "gpu" + str(row["gpu"]),
            "-c", str(root), "exec " + shlex.join(cmd)], check=True)
    command = ["env", "PYTHONPATH=" + str(root / "repo"), plan["python"], "-u", "-m",
        "medtoolbench.supplement.profile_oracle", "--plan", config["campaigns"][2], "--workers", str(config["oracle_workers"])]
    subprocess.run(["tmux", "new-window", "-t", session, "-n", "oracle-c4", "-c", str(root), "exec " + shlex.join(command)], check=True)
    write_json(path.parent / "launch.json", {"pipeline_hash": config["pipeline_hash"], "time": time.time(), "session": session,
        "queued_gpu_workers": len(plan["workers"]), "oracle_workers": config["oracle_workers"],
        "active_predecessor_interrupted": False})
    print(json.dumps({"session": session, "queued_gpu_workers": len(plan["workers"]), "oracle_workers": config["oracle_workers"]}), flush=True)


def finalize(path, config):
    with FileLease(path.parent / "locks/summary.lock"):
        if (path.parent / "complete.json").exists():
            return
        records = []
        for cp, expected in zip(config["campaigns"], config["plan_hashes"]):
            root = Path(cp).parent
            if not (root / "complete.json").exists() or not (root / "report/complete.json").exists():
                return
            completion = json.loads((root / "complete.json").read_text())
            report = json.loads((root / "report/complete.json").read_text())
            if not completion["all_pass"] or not report["all_pass"] or completion["plan_hash"] != expected or report["plan_hash"] != expected:
                raise RuntimeError("Invalid completion identity in corrected study")
            records.append({"campaign": cp, "plan_hash": expected, "report": str(root / "report/metrics.json"),
                "report_hash": digest((root / "report/metrics.json").read_text()), "episodes_test": report["episodes_test"]})
        write_json(path.parent / "complete.json", {"all_pass": True, "pipeline_hash": config["pipeline_hash"],
            "campaigns": records, "episodes_including_development": config["episodes"],
            "episodes_test": sum(r["episodes_test"] for r in records), "superseded_c3_validation_not_pooled": True, "time": time.time()})


def work(path, name):
    config, plan = load(path)
    row = validate_worker(plan, name)
    old_root = Path(config["predecessor_pipeline"]).parent
    status = path.parent / (name + ".json")
    def interrupted(signum, frame):
        raise KeyboardInterrupt("C repair continuation interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    with FileLease(path.parent / "locks" / (name + ".lock")):
        try:
            old_lease = None
            while old_lease is None:
                if (path.parent / "paused.json").exists():
                    raise RuntimeError("Repair continuation was paused")
                if predecessor_ready(old_root, name):
                    candidate = FileLease(old_root / "locks" / (name + ".lock"))
                    try:
                        candidate.__enter__()
                        old_lease = candidate
                        break
                    except BlockingIOError:
                        pass
                write_json(status, {"state": "waiting_for_predecessor_worker", "worker": row, "heartbeat": time.time(),
                    "predecessor": str(old_root / (name + ".json")), "gpu_memory_allocated_by_this_controller": False})
                time.sleep(15)
            try:
                load(path)
                validate_worker(plan, name)
                cp = Path(config["campaigns"][2])
                write_json(status, {"state": "executing_corrected_c", "worker": row, "heartbeat": time.time()})
                worker_main(cp, name)
                success = wait_campaign(cp.parent, status, [r["name"] for r in plan["workers"]])
                if success:
                    try:
                        with FileLease(cp.parent / "locks/statistical-report.lock"):
                            if not (cp.parent / "report/complete.json").exists():
                                from .report import generate
                                generate(cp)
                    except BlockingIOError:
                        pass
                write_json(status, {"state": "complete" if success else "corrected_c_failed", "worker": row, "heartbeat": time.time()})
                try:
                    finalize(path, config)
                except BlockingIOError:
                    pass
            finally:
                old_lease.__exit__()
        except BaseException:
            write_json(status, {"state": "failed_or_interrupted", "worker": row, "error": traceback.format_exc(), "heartbeat": time.time()})
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", type=Path, required=True)
    p.add_argument("--worker")
    p.add_argument("--launch", action="store_true")
    a = p.parse_args()
    if a.launch and not a.worker:
        launch(a.pipeline.resolve())
    elif a.worker and not a.launch:
        work(a.pipeline.resolve(), a.worker)
    else:
        raise RuntimeError("Choose --launch or --worker")


if __name__ == "__main__":
    main()
