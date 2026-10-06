"""Twelve raw operations for every policy: vanilla can exhaust the same ceiling."""
import argparse
import copy
import json
import time
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.runner import load_plan

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
source = load_plan(root / "data/supplement_a_v1/plan.json")
if a.out.exists():
    raise RuntimeError("Never overwrite a strict-budget freeze")
models = {"qwen35_9b", "ministral3_8b", "gemma3_12b"}
conditions = {"vanilla", "informed", "evidence_only", "medguard"}
jobs = []
for job in source["jobs"]:
    if job["arm"] not in {"qualification", "calls_24"} or job["model"] not in models:
        continue
    new = copy.deepcopy(job)
    new["id"] = "strict12--" + job["id"]
    new["arm"] = "strict_raw_budget_12"
    new["settings"]["raw_call_budget"] = 12
    if job["arm"] == "qualification":
        new["episodes"] = [e for e in new["episodes"] if e["condition"] in conditions]
    else:
        new["episodes"] = [dict(e, condition=c, key=e["task"] + "--" + e["hazard"] + "--" + c)
            for e in job["episodes"] if e["condition"] == "vanilla" for c in sorted(conditions)]
    if new["episodes"]:
        jobs.append(new)
plan = {k: copy.deepcopy(v) for k, v in source.items() if k not in {"plan_hash", "jobs", "source_hashes", "input_hashes"}}
plan.update(session="medtoolbench-supplement-d1", created_at=time.time(), jobs=jobs, require_oracle=False,
    qualification="Original fault assignments already passed all_v3; same runner with a strictly lower raw-call cap; fresh development audits required",
    scope="Sensitivity on the same fixed 50-task subset. Every policy can use all 12 raw calls within 12 turns. Equal hard ceilings do not imply equal realized token consumption.",
    remaining_separate_arms=[])
files = [Path(name) for name in source["source_hashes"] if not name.endswith("prepare_supplement.py")]
files.append(Path(__file__).resolve())
plan["source_hashes"] = {str(p): digest(p.read_text()) for p in files}
plan["input_hashes"] = {str(root / "data/supplement_a_v1/plan.json"): digest((root / "data/supplement_a_v1/plan.json").read_bytes().hex())}
plan["plan_hash"] = digest(plan)
a.out.mkdir(parents=True)
(a.out / "sources").mkdir()
for path in files:
    (a.out / "sources" / (digest(str(path))[:10] + "-" + path.name)).write_bytes(path.read_bytes())
write_json(a.out / "plan.json", plan)
write_json(a.out / "matrix.json", {"jobs": len(jobs), "episodes": sum(len(j["episodes"]) for j in jobs), "plan_hash": plan["plan_hash"]})
print((a.out / "matrix.json").read_text())
