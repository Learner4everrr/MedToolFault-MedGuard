"""Validate every new recoverable fault assignment without model/gold access."""
import argparse
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.build import save_gzip
from medtoolbench.research.contracts import grade
from medtoolbench.research.fhir import HapiProcess
from medtoolbench.research.job_state import write_json
from medtoolbench.research.oracle import public_task, solve
from .runner import load_plan, task_fixture
from .transport import FactorClient, StressProxy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    plan = load_plan(a.plan)
    root = a.plan.parent / "oracle"
    if root.exists():
        raise RuntimeError("Never overwrite oracle evidence")
    root.mkdir()
    tasks = {t["id"]: t for t in plan["tasks"]}
    unique = {}
    for job in plan["jobs"]:
        stress = job["settings"]["stress"]
        if not stress or stress.get("visibility_delay") or stress.get("concurrent"):
            continue
        for e in job["episodes"]:
            key = digest([e["task"], e["hazard"], stress])
            unique[key] = (e["task"], e["hazard"], stress)
    rows, results, lock = list(sorted(unique.items())), [], threading.Lock()
    write_json(root / "config.json", {"plan_hash": plan["plan_hash"], "assignments": len(rows),
        "original_assignments": "reuse original all_v3 clean/recovery validation without modifying it",
        "boundary_assignments": "explicitly not assumed recoverable", "model_inference": False})
    def worker(index):
        with HapiProcess(plan["image"], root / ("hapi-" + str(index)), heap="2g") as store:
            for key, (tid, hazard, stress) in rows[index::a.workers]:
                task = tasks[tid]
                before = store.reload(task_fixture(plan, task))
                proxy = StressProxy(store, task, hazard, stress)
                client = FactorClient(proxy.call, "medguard")
                def recover(method, path, body=None, actor="agent"):
                    response = None
                    for _ in range(8):
                        response = client.call(method, path, body)
                        if response["status"] < 500:
                            return response
                    return response
                answer = solve(public_task(task), recover, "vanilla")
                after = store.snapshot()
                g = grade(task, before, after, answer)
                if not g["success"]:
                    save_gzip(root / (key + "-failed.json.gz"), {"task": tid, "hazard": hazard, "stress": stress, "before": before,
                        "after": after, "trace": proxy.trace, "answer": answer, "grade": g})
                    raise RuntimeError("Recovery oracle failed: " + key)
                proof = {"key": key, "task": tid, "hazard": hazard, "stress": stress, "success": True,
                    "before_hash": digest(before), "after_hash": digest(after), "triggered": proxy.triggered,
                    "calls": len(proxy.trace), "answer": answer}
                with lock:
                    results.append(proof)
                    write_json(root / "progress.json", {"completed": len(results), "total": len(rows), "heartbeat": time.time()})
                save_gzip(root / (key + ".json.gz"), {**proof, "trace": proxy.trace, "grade": g})
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(worker, range(a.workers)))
        write_json(root / "complete.json", {"all_pass": len(results) == len(rows), "assignments": len(rows),
            "plan_hash": plan["plan_hash"], "model_inference": False, "results": results})
    except BaseException:
        write_json(root / "failed.json", {"error": traceback.format_exc(), "plan_hash": plan["plan_hash"]})
        raise


if __name__ == "__main__":
    main()
