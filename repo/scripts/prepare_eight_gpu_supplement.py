"""Audit a user-paused, development-only execution and rebind its frozen matrix."""
import argparse
import hashlib
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import FileLease, gpu_inventory, short_host, validate_worker
from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.pipeline import load
from medtoolbench.supplement.resource_pipeline import rebind, scientific_identity
from medtoolbench.supplement.runner import load_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plans", nargs=4, type=Path, required=True)
    p.add_argument("--slurm-job", type=int, required=True)
    a = p.parse_args()
    source, output = a.source.resolve(), a.out.resolve()
    config = load(source)
    if not json.loads((source.parent / "paused.json").read_text())["resume_requires_user_confirmation"]:
        raise RuntimeError("Expected the recorded user-requested pause")
    if output.exists() or any(path.exists() for path in a.plans):
        raise RuntimeError("Never overwrite an execution revision")
    inventory = gpu_inventory()
    if len(inventory) != 8 or {r["gpu"] for r in inventory} != set(range(8)):
        raise RuntimeError("Expected exactly eight currently visible GPUs")
    workers = [{"name": short_host() + "-gpu" + str(r["gpu"]), "host": short_host(), "gpu": r["gpu"],
        "uuid": r["uuid"], "slurm_job": a.slurm_job} for r in inventory]
    for row in workers:
        validate_worker({"workers": workers}, row["name"])
    old_plans = [load_plan(Path(path)) for path in config["campaigns"]]
    history = []
    with ExitStack() as locks:
        for worker in old_plans[0]["workers"]:
            locks.enter_context(FileLease(source.parent / "locks" / (worker["name"] + ".lock")))
        for name, plan in zip(config["campaigns"], old_plans):
            root = Path(name).parent
            for worker in plan["workers"]:
                locks.enter_context(FileLease(root / "locks" / (worker["name"] + ".lock")))
            job_by_id = {j["id"]: j for j in plan["jobs"]}
            for folder in (root / "jobs").glob("*"):
                if not folder.is_dir():
                    continue
                job = job_by_id[folder.name]
                locks.enter_context(FileLease(root / "locks" / (folder.name + ".lock")))
                if job["stage"] != "qualification" or (folder / "result.json").exists():
                    raise RuntimeError("Held-out or terminal shard exists; use an audited reuse handoff instead")
                for attempt in folder.glob("attempt*"):
                    if not attempt.is_dir():
                        continue
                    if (attempt / "audit.json").exists() or (attempt / "complete.json").exists():
                        raise RuntimeError("Completed or audit-invalid evidence cannot be silently restarted")
                    failure = json.loads((attempt / "failure.json").read_text())
                    if not failure["error"].endswith("KeyboardInterrupt: Received termination signal\n"):
                        raise RuntimeError("Unexpected interruption; inspect it before choosing recovery")
                    saved = list((attempt / "episodes").glob("*.json.gz"))
                    progress = json.loads((attempt / "progress.json").read_text()) if (attempt / "progress.json").exists() else {"completed": 0}
                    if len(saved) != progress["completed"]:
                        raise RuntimeError("Partial output/progress mismatch")
                    history.append({"run": str(attempt), "job": job["id"], "stage": job["stage"],
                        "saved_development_episodes": len(saved), "whole_shard_restart_reason": "Explicit user pause; no accuracy-based selection",
                        "evidence_sha256": {str(f.relative_to(attempt)): hashlib.sha256(f.read_bytes()).hexdigest()
                            for f in [*saved, attempt / "failure.json", attempt / "config.json"]}})
        output.mkdir(parents=True)
        (output / "sources").mkdir()
        new_plans = []
        for index, (parent, old_name, destination) in enumerate(zip(old_plans, config["campaigns"], a.plans)):
            destination = destination.resolve()
            letter = "abcd"[index]
            plan = rebind(parent, Path(old_name), workers, "medtoolbench-supplement-" + letter + "-8gpu")
            destination.mkdir(parents=True)
            write_json(destination / "plan.json", plan)
            write_json(destination / "matrix.json", {"jobs": len(plan["jobs"]),
                "episodes": sum(len(j["episodes"]) for j in plan["jobs"]), "plan_hash": plan["plan_hash"],
                "parent": old_name, "scientific_identity": scientific_identity(plan), "scientific_configuration_unchanged": True})
            new_plans.append(plan)
        files = [Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "medtoolbench/supplement/resource_pipeline.py"]
        hashes = dict(config["source_hashes"])
        hashes.update({str(f): digest(f.read_text()) for f in files})
        for name in hashes:
            f = Path(name)
            (output / "sources" / (digest(name)[:10] + "-" + f.name)).write_bytes(f.read_bytes())
        revised = {"created_at": time.time(), "campaigns": [str(path.resolve() / "plan.json") for path in a.plans],
            "plan_hashes": [plan["plan_hash"] for plan in new_plans], "source_hashes": hashes,
            "source_pipeline": str(source), "source_pipeline_hash": config["pipeline_hash"],
            "episodes": config["episodes"], "inference_api": False, "worker_count": 8,
            "session": "medtoolbench-supplement-8gpu", "resources": inventory, "slurm_job": a.slurm_job,
            "policy": "Eight allocated GPUs on one node; unchanged scientific matrix and full development/oracle gates; no remote node or model API",
            "oracles": [{"name": "oracle-a", "kind": "primary", "plan": str(a.plans[0].resolve() / "plan.json"), "workers": 4},
                {"name": "oracle-c", "kind": "actions", "plan": str(a.plans[2].resolve() / "plan.json"), "workers": 4}]}
        revised["pipeline_hash"] = digest(revised)
        write_json(output / "pipeline.json", revised)
        write_json(output / "handoff.json", {"time": time.time(), "user_authorized_resumption": True,
            "prior_shared_locks_acquired": True, "previous_heldout_jobs_started": 0, "completed_shards_discarded": 0,
            "saved_development_episodes_retained": sum(r["saved_development_episodes"] for r in history),
            "interrupted_attempts": history, "primary_study_changed": False, "experimental_evidence_deleted": False,
            "scientific_matrix_unchanged": True, "pipeline_hash": revised["pipeline_hash"]})
    print(json.dumps({"pipeline": str(output / "pipeline.json"), "pipeline_hash": revised["pipeline_hash"],
        "workers": len(workers), "episodes": revised["episodes"], "retained_interrupted_development": sum(r["saved_development_episodes"] for r in history)}))


if __name__ == "__main__":
    main()
