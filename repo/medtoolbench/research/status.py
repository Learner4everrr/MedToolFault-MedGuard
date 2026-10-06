"""Read-only experiment status; never confuse queued/development with held-out runs."""
import argparse
import json
import time
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return {"state": "legacy_non_atomic_file_being_updated"}


def status(data):
    result = {"time": time.time(), "development": {}, "primary": {"state": "not_frozen", "heldout_episodes_saved": 0},
              "whole_research_project_complete": False}
    qualifications = read(data / "primary_qualifications_v1.json") or {}
    for name, location in qualifications.items():
        root = Path(location)
        result["development"][name] = {"complete": (root / "complete.json").exists(),
                                       "progress": read(root / "progress.json"),
                                       "infrastructure_failure": (root / "infrastructure_failure.txt").exists()}
    result["preparation"] = read(data / "primary_preparation_v2/status.json")
    result["preparation_failure"] = read(data / "primary_preparation_v2/failed.json")
    root = data / "primary_study_v1"
    plan = read(root / "plan.json")
    if plan:
        complete, failed, running, waiting, episodes = [], [], [], [], 0
        for job in plan["jobs"]:
            folder = root / "jobs" / job["id"]
            row = read(folder / "result.json")
            if row and row.get("state") == "complete":
                complete.append(job["id"])
                episodes += row["episodes"]
            elif row and row.get("state") == "failed":
                failed.append(job["id"])
            else:
                attempts = sorted(folder.glob("attempt[0-9]*"))
                attempts = [a for a in attempts if a.is_dir()]
                if attempts:
                    progress = read(attempts[-1] / "progress.json") or {}
                    episodes += progress.get("completed", 0)
                    running.append({"job": job["id"], "attempt": attempts[-1].name, "progress": progress,
                                    "infrastructure_failure": (attempts[-1] / "infrastructure_failure.txt").exists()})
                else:
                    waiting.append(job["id"])
        result["primary"] = {"state": "complete" if (root / "complete.json").exists() else "incomplete",
                             "plan_hash": plan["plan_hash"], "expected_episodes": plan["expected_episodes"],
                             "heldout_episodes_saved": episodes, "audited_jobs_complete": len(complete),
                             "jobs_total": len(plan["jobs"]), "failed_jobs": failed, "active_or_interrupted_jobs": running,
                             "jobs_not_started": len(waiting), "report_complete": (root / "report/complete.json").exists(),
                             "workers": [read(root / f"worker{i}.json") for i in (0, 1)]}
    for name, location in (("full_oracle", "authentic_validation_all_v3"),
                           ("guard_boundary", "guard_boundary_v1"), ("external_largest_history", "external_largest_history_dev_v1"),
                           ("external_full_validation", "external_validation_all_v1")):
        proof = read(data / location / "complete.json")
        if proof:
            result[name] = {k: v for k, v in proof.items() if k not in {"results", "failures", "checks"}}
        else:
            result[name] = {"complete": False, "progress": read(data / location / "progress.json"),
                            "failure": read(data / location / "failed.json")}
    result["remaining_beyond_primary"] = ["external model evaluation", "model-based severity/compound and component ablations",
                                           "sampling/budget sensitivity", "portable environment", "measured manuscript/release materials"]
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("../data"))
    a = p.parse_args()
    print(json.dumps(status(a.data.resolve()), indent=2))
