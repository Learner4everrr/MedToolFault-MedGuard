"""Prepare an immutable resource deployment; never stop or launch experiments."""
import argparse
import json
import shutil
import time
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import deployment, extension_hashes
from medtoolbench.research.job_state import write_json
from medtoolbench.research.runtime import core_fingerprint
from medtoolbench.research.status import status
from medtoolbench.research.supervisor import load_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allocation", type=Path, required=True)
    a = p.parse_args()
    root, output = a.study.resolve(), a.out.resolve()
    if output.exists():
        raise RuntimeError("Deployment directory exists")
    plan = load_plan(root)
    if core_fingerprint() != plan["core_hashes"]:
        raise RuntimeError("Frozen core changed")
    allocations = json.loads(a.allocation.read_text())
    config = {"study": str(root), "created_at": time.time(), "plan_hash": plan["plan_hash"],
              "extension_hashes": extension_hashes(), "workers": allocations["workers"],
              "resource_evidence": allocations["resource_evidence"],
              "inference_python": str(root.parents[1] / "env_modern/bin/python"),
              "scientific_changes": [], "resource_change": "2 GPUs on aga38 -> 2 on aga38 + 4 on agb03",
              "lease_adapter": "run.GpuLease only; recorded in every new attempt placement file",
              "scheduling": "shared-filesystem exclusive job locks; no accuracy-based selection"}
    config["deployment_hash"] = digest(config)
    output.mkdir(parents=True)
    (output / "sources").mkdir()
    package = Path(__file__).resolve().parents[1] / "medtoolbench/research"
    for name in config["extension_hashes"]:
        shutil.copy2(package / name, output / "sources" / name)
    shutil.copy2(a.allocation, output / "allocation-source.json")
    write_json(output / "deployment.json", config)
    deployment(output / "deployment.json")
    write_json(output / "before-handoff-status.json", status(root.parent))
    print(json.dumps({"deployment": str(output / "deployment.json"), "hash": config["deployment_hash"],
                      "workers": len(config["workers"]), "existing_experiments_changed": False}))


if __name__ == "__main__":
    main()
