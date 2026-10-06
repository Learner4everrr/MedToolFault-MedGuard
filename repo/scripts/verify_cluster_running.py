"""Wait for real new output from every worker, then save a deployment audit."""
import argparse
import json
import time
from pathlib import Path

from cluster_status import read, summarize
from medtoolbench.research.cluster_execution import deployment
from medtoolbench.research.job_state import write_json


def inspect(path):
    config, plan = deployment(path)
    report = summarize(path)
    if report["failed_jobs"]:
        raise RuntimeError("A primary shard failed; inspect the recorded error")
    systems = {s["name"]: s for s in plan["systems"]}
    jobs = {j["id"]: j for j in plan["jobs"]}
    placements = []
    for row in report["workers"]:
        if not row["run"]:
            continue
        run = Path(row["run"])
        run_config = read(run / "config.json")
        model = read(run / "model.json")
        placement = read(run.parent / (run.name + ".placement.json"))
        if run_config:
            job = jobs[run.parent.name]
            system = systems[job["model"]]
            assert run_config["core_hashes"] == plan["core_hashes"]
            assert run_config["manifest_hash"] == plan["manifest_hash"]
            assert run_config["split"] == "test"
            assert run_config["conditions"] == plan["conditions"]
            assert run_config["batch_size"] == system["batch_size"]
            assert run_config["thinking"] == system["thinking"]
            assert Path(run_config["model"]).resolve() == Path(system["path"]).resolve()
            assert run_config["task_shard"] == job["task_shard"]
            assert all(run_config[k] == v for k, v in plan["generation"].items())
        if placement:
            assert placement["worker"]["name"] == row["name"]
            assert placement["deployment_hash"] == config["deployment_hash"]
        row["model_loaded"] = bool(model)
        row["scientific_configuration_verified"] = bool(run_config and placement)
        if model:
            assert model["torch"] == "2.8.0+cu128" and model["transformers"] == "5.17.0"
            assert model["gpu"] == "NVIDIA A100-SXM4-40GB"
            placements.append({"worker": row["name"], "model": model, "placement": placement})
    outputs = [r["run"] for r in report["workers"] if r["run"]]
    if len(outputs) != len(set(outputs)):
        raise RuntimeError("Two workers report the same active run")
    ready = all(w["model_loaded"] and w["scientific_configuration_verified"]
                and w["newly_generated"] > 0 and w["heartbeat_age_seconds"] < 30
                for w in report["workers"]) if len(outputs) == len(config["workers"]) else False
    report.update(all_workers_producing_new_output=ready, placements=placements)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--deployment", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=1800)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError("Verification output exists")
    deadline, previous = time.monotonic() + a.timeout, None
    while True:
        report = inspect(a.deployment.resolve())
        summary = {"saved": report["saved_episodes"], "workers": [
            {k: w[k] for k in ("name", "state", "model_loaded", "recovered", "newly_generated")}
            for w in report["workers"]]}
        if summary != previous:
            print(json.dumps(summary), flush=True)
            previous = summary
        if report["all_workers_producing_new_output"]:
            write_json(a.out, report)
            print(json.dumps({"all_workers_producing_new_output": True,
                              "saved": report["saved_episodes"], "report": str(a.out)}), flush=True)
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Not all six workers produced new output within the verification window")
        time.sleep(15)
