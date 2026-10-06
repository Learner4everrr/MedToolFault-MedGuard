"""Prospective V2 correction before held-out supplemental execution."""
import argparse
import copy
import json
import time
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.runner import load_plan

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
old = load_plan(a.source.resolve())
if a.out.exists():
    raise RuntimeError("Never overwrite revised experiment")
plan = copy.deepcopy(old)
plan.pop("plan_hash")
plan.update(version=2, created_at=time.time(), supersedes_plan=str(a.source.resolve()),
    supersedes_plan_hash=old["plan_hash"], runner_module="medtoolbench.supplement.corrected_runner",
    correction="Capability metadata reflects persistent interface drift even on non-injected /metadata requests. All V1 evidence retained; V2 reruns development and full oracle qualification.")
files = [Path(__file__).resolve()]
files += [Path(old["project"]) / "repo/medtoolbench/supplement" / (n + ".py") for n in ("corrected_runner", "corrected_oracle")]
plan["source_hashes"].update({str(p): digest(p.read_text()) for p in files})
plan["session"] = old["session"][:-1] + "2"
plan["plan_hash"] = digest(plan)
a.out.mkdir(parents=True)
(a.out / "sources").mkdir()
for name in plan["source_hashes"]:
    path = Path(name)
    (a.out / "sources" / (digest(str(path))[:10] + "-" + path.name)).write_bytes(path.read_bytes())
write_json(a.out / "plan.json", plan)
write_json(a.out / "matrix.json", {"jobs": len(plan["jobs"]), "episodes": sum(len(j["episodes"]) for j in plan["jobs"]),
    "plan_hash": plan["plan_hash"], "supersedes": str(a.source.resolve())})
print((a.out / "matrix.json").read_text())
