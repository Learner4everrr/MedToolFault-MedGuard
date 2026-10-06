"""Record canonical A/B/C4/D results and noninterrupting C4 continuation."""
import argparse
import copy
import json
import time
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.resource_pipeline import load
from medtoolbench.supplement.profile_plan import validate_repair
from medtoolbench.supplement.runner import load_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--corrected-c", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    previous, plans = load(a.source.resolve())
    repaired = load_plan(a.corrected_c.resolve())
    validate_repair(plans[2], repaired)
    preflight = json.loads(a.preflight.read_text())
    if not preflight["all_pass"] or preflight["plan_hash"] != repaired["plan_hash"]:
        raise RuntimeError("Missing successful C4 preflight")
    if a.out.exists():
        raise RuntimeError("Never overwrite a repair continuation")
    config = copy.deepcopy(previous)
    config.pop("pipeline_hash")
    config.pop("oracles")
    config.update(created_at=time.time(), predecessor_pipeline=str(a.source.resolve()),
        predecessor_pipeline_hash=previous["pipeline_hash"], session="medtoolbench-c4-recovery",
        preflight=str(a.preflight.resolve()), oracle_workers=12,
        policy="Leave active A/B/D unchanged; CPU-validate C4 now; each corrected C worker waits for predecessor terminal state and released worker lock",
        execution_order="A then B then D using unchanged predecessor controllers; corrected C follows on released GPUs",
        superseded_c_diagnostic=previous["campaigns"][2])
    config["campaigns"][2] = str(a.corrected_c.resolve())
    config["plan_hashes"][2] = repaired["plan_hash"]
    root = Path(repaired["project"]) / "repo"
    files = [Path(__file__).resolve(), root / "medtoolbench/supplement/c_repair_pipeline.py",
        root / "scripts/validate_profile_repair.py", root / "medtoolbench/supplement/profile_plan.py"]
    config["source_hashes"].update({str(f): digest(f.read_text()) for f in files})
    config["pipeline_hash"] = digest(config)
    a.out.mkdir(parents=True)
    (a.out / "sources").mkdir()
    for name in config["source_hashes"]:
        f = Path(name)
        (a.out / "sources" / (digest(name)[:10] + "-" + f.name)).write_bytes(f.read_bytes())
    write_json(a.out / "pipeline.json", config)
    print(json.dumps({"pipeline_hash": config["pipeline_hash"], "episodes": config["episodes"],
        "active_A_B_D_unchanged": True, "corrected_c": config["campaigns"][2]}), flush=True)


if __name__ == "__main__":
    main()
