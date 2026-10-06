"""Freeze the isolated C4 metadata import correction before any C model run."""
import argparse
import copy
import json
import time
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.profile_plan import experiment_identity, validate_repair
from medtoolbench.supplement.runner import load_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    source, out = a.source.resolve(), a.out.resolve()
    parent = load_plan(source)
    if out.exists():
        raise RuntimeError("Never overwrite a correction or its evidence")
    if list((source.parent / "jobs").glob("*/attempt*")) or list((source.parent / "jobs").glob("*/result.json")):
        raise RuntimeError("C model evidence already exists; explicit recovery review required")
    if not (source.parent / "oracle/failed.json").exists():
        raise RuntimeError("Expected the preserved C3 validation failure")
    revised = copy.deepcopy(parent)
    revised.pop("plan_hash")
    revised.pop("resource_handoff", None)
    revised.update(version=4, created_at=time.time(), session="medtoolbench-supplement-c4",
        runner_module="medtoolbench.supplement.profile_runner",
        profile_correction={"parent_plan": str(source), "parent_plan_hash": parent["plan_hash"],
            "experiment_identity": experiment_identity(parent),
            "cause": "Fixture import removed all metadata but strict roundtrip checks retain meta.profile",
            "change": "Preserve non-server-owned metadata during C fixture import; original grader and state comparison retained",
            "model_evaluation_previously_started": False, "prior_failure_evidence_retained": True})
    files = [Path(__file__).resolve()] + [Path(parent["project"]) / "repo/medtoolbench/supplement" / (name + ".py")
        for name in ("profile_store", "profile_runner", "profile_oracle", "profile_plan")]
    revised["source_hashes"].update({str(f): digest(f.read_text()) for f in files})
    revised["plan_hash"] = digest(revised)
    validate_repair(parent, revised)
    out.mkdir(parents=True)
    (out / "sources").mkdir()
    for name in revised["source_hashes"]:
        f = Path(name)
        (out / "sources" / (digest(name)[:10] + "-" + f.name)).write_bytes(f.read_bytes())
    write_json(out / "plan.json", revised)
    record = {"plan_hash": revised["plan_hash"], "parent_plan_hash": parent["plan_hash"],
        "jobs": len(revised["jobs"]), "episodes": sum(len(j["episodes"]) for j in revised["jobs"]),
        "tasks": len(revised["tasks"]), "oracle_checks": sum(len(t["hazards"]) for t in revised["tasks"]),
        "experiment_design_unchanged": True, "original_source_hashes_preserved": True}
    write_json(out / "matrix.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
