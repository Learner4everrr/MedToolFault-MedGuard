"""Real-HAPI regression on deterministic representative C4 fixtures and hazards."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.contracts import grade
from medtoolbench.research.fhir import HapiStore, normalized_state
from medtoolbench.research.job_state import write_json
from medtoolbench.research.oracle import public_task, solve
from medtoolbench.supplement.action_oracle import external_solve
from medtoolbench.supplement.corrected_runner import InterfaceConsistentProxy
from medtoolbench.supplement.profile_store import ProfilePreservingProcess
from medtoolbench.supplement.runner import load_plan, task_fixture
from medtoolbench.supplement.transport import FactorClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    plan = load_plan(a.plan)
    if a.out.exists():
        raise RuntimeError("Never overwrite preflight evidence")
    a.out.mkdir(parents=True)
    sizes = {t["id"]: len(task_fixture(plan, t)) for t in plan["tasks"]}
    external = [t for t in plan["tasks"] if t["template"] == 101]
    original = [t for t in plan["tasks"] if t["template"] != 101]
    first = external[0]
    chosen = [first, *[t for t in external if t["parent_task"] == first["parent_task"]],
        max(external, key=lambda t: sizes[t["id"]]), max(original, key=lambda t: sizes[t["id"]]),
        min(original, key=lambda t: sizes[t["id"]])]
    tasks = list({t["id"]: t for t in chosen}.values())
    write_json(a.out / "config.json", {"plan_hash": plan["plan_hash"], "tasks": [t["id"] for t in tasks],
        "selection": "first failing external task and paired branch, largest external, largest/smallest primary fixtures",
        "model_inference": False, "independent_of_model_outcomes": True})
    def work(t):
        fixture = task_fixture(plan, t)
        proofs = []
        with ProfilePreservingProcess(plan["image"], a.out / t["id"], heap="2g") as store:
            reproduced = None
            if t["id"] == first["id"]:
                try:
                    HapiStore.reload(store, fixture)
                    reproduced = False
                except AssertionError:
                    reproduced = True
                if not reproduced:
                    raise RuntimeError("Expected original regression was not reproduced")
            for hazard in t["hazards"]:
                before = store.reload(fixture)
                proxy = InterfaceConsistentProxy(store, t, hazard)
                client = FactorClient(proxy.call, "medguard")
                answer = external_solve(public_task(t), client.call) if t["template"] == 101 else solve(public_task(t), client.call, "vanilla")
                g = grade(t, before, store.snapshot(), answer)
                if not g["success"]:
                    raise RuntimeError("Preflight recovery failed: " + t["id"] + " / " + hazard)
                store.reset()
                if normalized_state(store.snapshot()) != normalized_state(fixture):
                    raise RuntimeError("Strict reset equality failed")
                proofs.append({"hazard": hazard, "strict_roundtrip": True, "recovery_success": True, "triggered": proxy.triggered})
        result = {"task": t["id"], "fixture_resources": len(fixture), "fixture_hash": digest(fixture),
            "original_failure_reproduced": reproduced, "checks": proofs}
        write_json(a.out / (t["id"] + ".json"), result)
        print(json.dumps({"task": t["id"], "checks_passed": len(proofs)}), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, tasks))
    report = {"all_pass": True, "plan_hash": plan["plan_hash"], "tasks": len(results),
        "hazard_checks": sum(len(r["checks"]) for r in results), "results": results, "time": time.time(), "model_inference": False}
    write_json(a.out / "complete.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
