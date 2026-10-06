"""Resource-only eight-GPU continuation; frozen scientific runners are unchanged."""
import argparse
import copy
import json
import os
import shlex
import signal
import subprocess
import time
import traceback
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import AllocatedGpuLease, FileLease, validate_worker
from medtoolbench.research.job_state import write_json
from .campaign import worker_main
from .pipeline import load as load_base
from .runner import load_plan

RESOURCE_FIELDS = {"workers", "session", "created_at", "resource_handoff", "plan_hash"}


def scientific_identity(plan):
    return digest({k: v for k, v in plan.items() if k not in RESOURCE_FIELDS})


def rebind(parent, parent_path, workers, session):
    plan = copy.deepcopy(parent)
    plan.pop("plan_hash")
    plan.update(workers=copy.deepcopy(workers), session=session, created_at=time.time(),
        resource_handoff={"parent_plan": str(parent_path.resolve()), "parent_plan_hash": parent["plan_hash"],
            "reason": "User paused before held-out execution, replaced two-node allocation with eight GPUs on one node",
            "partial_development_policy": "Preserve interrupted evidence; repeat whole development shards in a new execution revision",
            "scientific_identity": scientific_identity(parent)})
    if scientific_identity(plan) != scientific_identity(parent):
        raise RuntimeError("Resource-only rebind changed scientific configuration")
    plan["plan_hash"] = digest(plan)
    return plan


def load(path):
    config = load_base(path)
    plans = [load_plan(Path(p)) for p in config["campaigns"]]
    workers = plans[0]["workers"]
    if len(workers) != 8 or len({w["name"] for w in workers}) != 8 or len({w["uuid"] for w in workers}) != 8:
        raise RuntimeError("Exactly eight unique GPU workers are required")
    if {w["gpu"] for w in workers} != set(range(8)) or len({w["host"] for w in workers}) != 1:
        raise RuntimeError("Expected eight GPUs on a single declared node")
    if len({w["slurm_job"] for w in workers}) != 1 or config["worker_count"] != 8:
        raise RuntimeError("Resource manifest and allocation disagree")
    for plan in plans:
        handoff = plan["resource_handoff"]
        parent = load_plan(Path(handoff["parent_plan"]))
        if parent["plan_hash"] != handoff["parent_plan_hash"] or plan["workers"] != workers:
            raise RuntimeError("Parent plan or shared allocation changed")
        if scientific_identity(parent) != scientific_identity(plan) or scientific_identity(plan) != handoff["scientific_identity"]:
            raise RuntimeError("Scientific task/model/budget/grader configuration changed")
    if (path.parent / "paused.json").exists():
        raise RuntimeError("This execution revision is paused; explicit recovery is required")
    return config, plans


def terminal(root, names):
    if (root / "complete.json").exists():
        return True
    if (root / "failed.json").exists():
        return False
    statuses = [root / "workers" / (n + ".json") for n in names]
    if all(p.exists() for p in statuses) and all(json.loads(p.read_text()).get("state") == "qualification_failed" for p in statuses):
        return False
    return None


def wait_campaign(root, status, names):
    while True:
        outcome = terminal(root, names)
        if outcome is not None:
            return outcome
        write_json(status, {"state": "waiting_for_campaign", "campaign": str(root), "heartbeat": time.time()})
        time.sleep(15)


def probe(path, name):
    config, plans = load(path)
    row = validate_worker(plans[0], name)
    out = path.parent / "probes" / (name + ".json")
    if out.exists():
        raise RuntimeError("Do not overwrite a CUDA probe")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(row["gpu"])
    with AllocatedGpuLease(row):
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Worker must see exactly one CUDA device")
        device = torch.cuda.get_device_properties(0)
        actual = str(device.uuid).removeprefix("GPU-")
        if actual != row["uuid"].removeprefix("GPU-"):
            raise RuntimeError("CUDA device disagrees with recorded physical UUID")
        values = torch.arange(8, dtype=torch.float32, device="cuda:0")
        if (values * values).sum().item() != 140:
            raise RuntimeError("CUDA arithmetic probe failed")
        out.parent.mkdir(exist_ok=True)
        write_json(out, {"all_pass": True, "worker": row, "pipeline_hash": config["pipeline_hash"],
            "torch": torch.__version__, "gpu": device.name, "cuda_uuid": actual, "time": time.time(),
            "model_inference": False, "counts_toward_results": False})
    print(json.dumps({"probe": name, "all_pass": True}), flush=True)


def launch(path):
    config, plans = load(path)
    first = plans[0]
    rows = first["workers"]
    for row in rows:
        validate_worker(first, row["name"])
        proof = json.loads((path.parent / "probes" / (row["name"] + ".json")).read_text())
        if not proof["all_pass"] or proof["pipeline_hash"] != config["pipeline_hash"] or proof["worker"] != row:
            raise RuntimeError("Eight valid GPU probes are required before launch")
    session = config["session"]
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError("Execution session already exists; inspect it")
    root = Path(first["project"])
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "control", "-c", str(root)], check=True)
    subprocess.run(["tmux", "set-option", "-t", session, "remain-on-exit", "on"], check=True)
    for row in rows:
        cmd = ["env", "PYTHONPATH=" + str(root / "repo"), first["python"], "-u", "-m",
            "medtoolbench.supplement.resource_pipeline", "--pipeline", str(path), "--worker", row["name"]]
        subprocess.run(["tmux", "new-window", "-t", session, "-n", "gpu" + str(row["gpu"]),
            "-c", str(root), "exec " + shlex.join(cmd)], check=True)
    for oracle in config["oracles"]:
        cmd = ["env", "PYTHONPATH=" + str(root / "repo"), first["python"], "-u", "-m",
            "medtoolbench.supplement.corrected_oracle", "--kind", oracle["kind"], "--plan", oracle["plan"],
            "--workers", str(oracle["workers"])]
        subprocess.run(["tmux", "new-window", "-t", session, "-n", oracle["name"],
            "-c", str(root), "exec " + shlex.join(cmd)], check=True)
    write_json(path.parent / "launch.json", {"time": time.time(), "session": session, "workers": rows,
        "oracles": config["oracles"], "pipeline_hash": config["pipeline_hash"]})
    print(json.dumps({"workers_launched": len(rows), "session": session, "oracles_launched": len(config["oracles"])}), flush=True)


def work(path, name):
    config, plans = load(path)
    row = validate_worker(plans[0], name)
    names = [r["name"] for r in plans[0]["workers"]]
    status = path.parent / (name + ".json")
    outcomes = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Resource worker received termination signal")
    signal.signal(signal.SIGTERM, interrupted)
    with FileLease(path.parent / "locks" / (name + ".lock")):
        try:
            for campaign in config["campaigns"]:
                load(path)
                cp = Path(campaign)
                write_json(status, {"state": "executing_campaign", "campaign": campaign, "worker": row, "heartbeat": time.time()})
                # Unchanged scientific queue, model adapter, batches, seeds and audits.
                worker_main(cp, name)
                complete = wait_campaign(cp.parent, status, names)
                outcomes.append({"campaign": campaign, "all_pass": complete})
                if complete:
                    try:
                        with FileLease(cp.parent / "locks" / "statistical-report.lock"):
                            if not (cp.parent / "report/complete.json").exists():
                                from .report import generate
                                write_json(status, {"state": "reporting", "campaign": campaign, "heartbeat": time.time()})
                                generate(cp)
                    except BlockingIOError:
                        pass
            write_json(status, {"state": "complete" if all(r["all_pass"] for r in outcomes) else "completed_with_failures",
                "worker": row, "campaigns": outcomes, "heartbeat": time.time()})
            try:
                with FileLease(path.parent / "locks/finish.lock"):
                    if all((Path(cp).parent / "complete.json").exists() and (Path(cp).parent / "report/complete.json").exists() for cp in config["campaigns"]):
                        write_json(path.parent / "complete.json", {"all_pass": True, "pipeline_hash": config["pipeline_hash"],
                            "episodes_including_development": config["episodes"], "time": time.time()})
            except BlockingIOError:
                pass
        except BaseException:
            write_json(status, {"state": "interrupted_or_failed", "worker": row, "error": traceback.format_exc(),
                "campaigns": outcomes, "heartbeat": time.time()})
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", type=Path, required=True)
    p.add_argument("--worker")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--launch", action="store_true")
    a = p.parse_args()
    path = a.pipeline.resolve()
    if a.launch and not a.worker and not a.probe:
        launch(path)
    elif a.worker and not a.launch:
        (probe if a.probe else work)(path, a.worker)
    else:
        raise RuntimeError("Choose --launch or --worker [--probe]")


if __name__ == "__main__":
    main()
