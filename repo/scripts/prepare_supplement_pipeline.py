import argparse
import json
import time
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.runner import load_plan

p = argparse.ArgumentParser()
p.add_argument("--plans", nargs="+", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
if a.out.exists():
    raise RuntimeError("Never overwrite the continuation manifest")
plans = [load_plan(path.resolve()) for path in a.plans]
root = Path(plans[0]["project"])
files = [Path(__file__).resolve()]
files += [root / "repo/medtoolbench/supplement" / (n + ".py") for n in ("pipeline", "report", "campaign", "runner")]
config = {"created_at": time.time(), "campaigns": [str(p.resolve()) for p in a.plans],
    "plan_hashes": [p["plan_hash"] for p in plans], "source_hashes": {str(p): digest(p.read_text()) for p in files},
    "episodes": sum(len(j["episodes"]) for p in plans for j in p["jobs"]),
    "inference_api": False, "policy": "Use exactly the six declared allocated GPUs; never modify prior scientific plans; every campaign has independent development/audit gates"}
config["pipeline_hash"] = digest(config)
a.out.mkdir(parents=True)
(a.out / "sources").mkdir()
for path in files:
    (a.out / "sources" / (digest(str(path))[:10] + "-" + path.name)).write_bytes(path.read_bytes())
write_json(a.out / "pipeline.json", config)
print(json.dumps(config, indent=2))
