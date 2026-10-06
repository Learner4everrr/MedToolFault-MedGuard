"""Freeze new action tasks and baseline/guard evaluation, separately from primary."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.job_state import write_json
from medtoolbench.research.runtime import core_fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    primary = json.loads((root / "data/primary_study_v1/plan.json").read_text())
    tasks = json.loads(a.tasks.read_text())
    if a.out.exists():
        raise RuntimeError("Never overwrite an action freeze")
    systems = [s for s in primary["systems"] if s["name"] in {"qwen35_9b", "ministral3_8b", "gemma3_12b"}]
    jobs = []
    settings = {"seed": 20260917, "max_turns": 12, "max_new_tokens": 2048, "max_context": 16384,
        "raw_call_budget": 48, "temperature": 0, "stress": {}}
    arms = sorted({t["arm"] for t in tasks})
    for stage in ("qualification", "test"):
        for arm in arms:
            eligible = [t for t in tasks if t["arm"] == arm and t["split"] == ("dev" if stage == "qualification" else "test")]
            if stage == "qualification":
                eligible = sorted(eligible, key=lambda t: t["id"])[:2]
            entries = [{"task": t["id"], "hazard": h, "condition": c, "key": t["id"] + "--" + h + "--" + c}
                for t in eligible for h in t["hazards"] for c in ("vanilla", "medguard")]
            for s in systems:
                for i in range(2 if stage == "qualification" else 8):
                    block = entries[i::(2 if stage == "qualification" else 8)]
                    if block:
                        jobs.append({"id": stage + "--" + arm + "--" + s["name"] + "--" + str(i),
                            "arm": arm, "stage": stage, "model": s["name"], "settings": settings, "episodes": block})
    files = [Path(__file__).resolve(), root / "repo/scripts/prepare_action_supplement.py"]
    files += [root / "repo/medtoolbench/supplement" / (n + ".py") for n in ("runner", "campaign", "transport", "action_oracle")]
    files += [root / "repo/medtoolbench/research" / (n + ".py") for n in ("cluster_execution", "cluster_supervisor", "audit", "supervisor")]
    plan = {"version": 1, "project": str(root), "session": "medtoolbench-supplement-c1", "created_at": time.time(),
        "python": str(root / "env_modern/bin/python"), "benchmark": primary["benchmark"], "image": primary["image"],
        "primary_core_hashes": core_fingerprint(), "source_hashes": {str(p): digest(p.read_text()) for p in files},
        "input_hashes": {str(a.tasks.resolve()): digest(a.tasks.read_bytes().hex())}, "require_oracle": True,
        "workers": json.loads((root / "data/primary_study_v1/execution_sixgpu_v1/deployment.json").read_text())["workers"],
        "systems": systems, "tasks": tasks, "jobs": jobs, "inference_api": False,
        "scope": "Semisynthetic branch and write-attribution experiments. External patient source does not make operational rules authentic clinical decisions.",
        "statistics": {"bootstrap_draws": 10000, "cluster": "original patient/group", "weighting": "average branches/faults within parent task",
            "endpoints": ["safe success", "unsafe state", "unnecessary action", "duplicate creation"],
            "no_claims": ["additional independent patients from replicas", "clinical harm rates", "treatment-policy validity", "preregistered primary evidence"]}}
    plan["plan_hash"] = digest(plan)
    a.out.mkdir(parents=True)
    (a.out / "sources").mkdir()
    for path in files:
        (a.out / "sources" / (digest(str(path))[:10] + "-" + path.name)).write_bytes(path.read_bytes())
    write_json(a.out / "plan.json", plan)
    counts = Counter()
    for job in jobs:
        counts[job["arm"]] += len(job["episodes"])
    write_json(a.out / "matrix.json", {"jobs": len(jobs), "episodes": sum(counts.values()), "arms": counts, "plan_hash": plan["plan_hash"]})
    print((a.out / "matrix.json").read_text())


if __name__ == "__main__":
    main()
