"""Independent online workflows for semi-synthetic task qualification."""
import argparse
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

from medtoolbench.storage import digest
from medtoolbench.research.build import save_gzip
from medtoolbench.research.contracts import NOW, grade, service
from medtoolbench.research.fhir import HapiProcess
from medtoolbench.research.job_state import write_json
from medtoolbench.research.oracle import public_task, solve
from .runner import load_plan, task_fixture
from .transport import FactorClient, StressProxy


def external_solve(public, call):
    params, patient = public["parameters"], public["patient"]
    query = urlencode({"patient": patient, "code": params["code"], "status": "final,amended,corrected", "date": "le" + NOW, "_sort": "-date", "_count": 1})
    response = call("GET", "/Observation?" + query)
    if response["status"] != 200:
        raise RuntimeError(response)
    rows = [e["resource"] for e in response["body"].get("entry", [])]
    if not rows:
        raise RuntimeError("No valid source measurement returned")
    value = rows[0]["valueQuantity"]["value"]
    act = value < params["threshold"] if params["comparison"] == "lt" else value >= params["threshold"]
    if act:
        response = call("POST", "/ServiceRequest", service(patient, params["code"], params["system"]))
        if response["status"] >= 400:
            raise RuntimeError(response)
    return {"completed": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    plan = load_plan(a.plan)
    root = a.plan.parent / "oracle"
    if root.exists():
        raise RuntimeError("Never overwrite action validation")
    root.mkdir()
    tasks = {t["id"]: t for t in plan["tasks"]}
    work = [(t["id"], h) for t in plan["tasks"] for h in t["hazards"]]
    reports, lock = [], threading.Lock()
    write_json(root / "config.json", {"plan_hash": plan["plan_hash"], "checks": len(work), "model_inference": False})
    def worker(i):
        with HapiProcess(plan["image"], root / ("hapi-" + str(i)), heap="2g") as store:
            for tid, hazard in work[i::a.workers]:
                t = tasks[tid]
                before = store.reload(task_fixture(plan, t))
                proxy = StressProxy(store, t, hazard)
                client = FactorClient(proxy.call, "medguard")
                answer = external_solve(public_task(t), client.call) if t["template"] == 101 else solve(public_task(t), client.call, "vanilla")
                after = store.snapshot()
                g = grade(t, before, after, answer)
                key = tid + "--" + hazard
                save_gzip(root / (key + ".json.gz"), {"task": tid, "hazard": hazard, "trace": proxy.trace, "grade": g})
                if not g["success"]:
                    raise RuntimeError("Action oracle failed: " + key)
                with lock:
                    reports.append({"key": key, "success": True, "triggered": proxy.triggered})
                    write_json(root / "progress.json", {"completed": len(reports), "total": len(work), "heartbeat": time.time()})
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(worker, range(a.workers)))
        write_json(root / "complete.json", {"all_pass": len(reports) == len(work), "checks": len(work),
            "plan_hash": plan["plan_hash"], "results": reports, "model_inference": False})
    except BaseException:
        write_json(root / "failed.json", {"error": traceback.format_exc(), "plan_hash": plan["plan_hash"]})
        raise


if __name__ == "__main__":
    main()
