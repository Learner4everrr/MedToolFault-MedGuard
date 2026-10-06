"""Six-GPU status and a historical, model-weighted primary-only ETA."""
import argparse
import heapq
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from medtoolbench.research.cluster_execution import deployment
from medtoolbench.research.job_state import write_json


def read(path, fallback=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return fallback


def retained_completed(run):
    """Count preserved source batches even while a restart is re-auditing them."""
    progress = read(run / "progress.json", {})
    recovery = read(run / "recovery.json", {})
    count = max(progress.get("completed", 0), len(recovery.get("recovered_episode_keys", [])))
    config = read(run / "config.json", {})
    if config.get("resume_from"):
        source = Path(config["resume_from"])
        proof = read(source / "audit.json")
        old = read(source / "config.json", {})
        if (proof and not proof["errors"] and not proof["unexpected"]
                and old.get("core_hashes") == config.get("core_hashes")
                and old.get("expected_keys") == config.get("expected_keys")):
            count = max(count, read(source / "progress.json", {}).get("completed", 0))
    return count


def summarize(config_path):
    config, plan = deployment(config_path)
    root, control = Path(config["study"]), config_path.parent
    rates, pending, done, failed, saved = defaultdict(list), [], [], [], 0
    for job in plan["jobs"]:
        folder = root / "jobs" / job["id"]
        result = read(folder / "result.json")
        if result and result["state"] == "complete":
            done.append(job["id"])
            saved += result["episodes"]
            run = Path(result["run"])
            completion = read(run / "complete.json")
            # Resumed wall times exclude the historical source batches.
            recovered = len(read(run / "recovery.json", {}).get("recovered_episode_keys", []))
            generated = completion["completed"] - recovered
            if generated > 0:
                rates[job["model"]].append(completion["seconds"] / generated)
            continue
        if result and result["state"] == "failed":
            failed.append({"job": job["id"], "error": result.get("error")})
        attempts = sorted((p for p in folder.glob("attempt*") if p.is_dir() and p.name[7:].isdigit()),
                          key=lambda p: int(p.name[7:]))
        complete = 0
        if attempts:
            run = attempts[-1]
            complete = retained_completed(run)
        saved += complete
        pending.append({"job": job["id"], "model": job["model"],
                        "remaining_episodes": job["expected_episodes"] - complete})
    workers = []
    for row in config["workers"]:
        state = read(control / (row["name"] + ".json"), {"state": "not_started"})
        output = Path(state["out"]) if state.get("out") else None
        progress = read(output / "progress.json", {}) if output else {}
        recovery = read(output / "recovery.json", {}) if output else {}
        recovered = len(recovery.get("recovered_episode_keys", []))
        workers.append({"name": row["name"], "state": state["state"],
                        "heartbeat_age_seconds": round(time.time() - state["heartbeat"], 1) if "heartbeat" in state else None,
                        "pid": state.get("pid"), "run": str(output) if output else None,
                        "completed": progress.get("completed", recovered), "recovered": recovered,
                        "retained_episodes_including_source": retained_completed(output) if output else 0,
                        "newly_generated": max(0, progress.get("completed", recovered) - recovered),
                        "total": progress.get("total"), "run_seconds": progress.get("seconds")})
    medians = {m: statistics.median(values) for m, values in rates.items()}
    estimates = [dict(j, estimated_gpu_seconds=j["remaining_episodes"] * medians[j["model"]]) for j in pending]
    # Preserve already running assignments; distribute remaining jobs in the
    # frozen order to the next available worker. It is an estimate, not a SLA.
    by_job = {j["job"]: j for j in estimates}
    active = {Path(w["run"]).parent.name for w in workers if w["run"] and w["state"] == "running"}
    loads = [by_job[j]["estimated_gpu_seconds"] for j in active if j in by_job]
    loads += [0.0] * (len(config["workers"]) - len(loads))
    heapq.heapify(loads)
    for item in estimates:
        if item["job"] not in active:
            heapq.heappush(loads, heapq.heappop(loads) + item["estimated_gpu_seconds"])
    return {"captured_at": time.time(), "saved_episodes": saved, "expected_episodes": plan["expected_episodes"],
            "percent": round(saved / plan["expected_episodes"] * 100, 2), "audited_jobs": len(done),
            "jobs_total": len(plan["jobs"]), "failed_jobs": failed, "workers": workers,
            "historical_model_seconds_per_episode_median": medians,
            "estimated_remaining_gpu_hours": sum(j["estimated_gpu_seconds"] for j in estimates) / 3600,
            "estimated_six_worker_wall_hours": max(loads, default=0) / 3600,
            "eta_scope": "Frozen primary matrix only; historical model-weighted estimate, excludes reporting and other study arms",
            "remaining_jobs": estimates, "plan_hash": plan["plan_hash"],
            "legacy_controller": "intentionally superseded; retain its interruption records",
            "whole_research_project_complete": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--deployment", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    report = summarize(a.deployment.resolve())
    if a.out:
        if a.out.exists():
            raise RuntimeError("Status snapshot exists")
        write_json(a.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "remaining_jobs"}, indent=2))
