"""Validate frozen task contracts against real owned HAPI, no model inference."""
import argparse
import json
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medtoolbench.storage import digest
from .build import load_fixture, save_gzip
from .contracts import grade
from .fhir import HapiProcess
from .oracle import public_task, solve
from .transport import Proxy
from .runtime import core_fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-template", type=int)
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        raise SystemExit("Use 1–8 owned HAPI workers")
    if a.out.exists():
        raise SystemExit("Use a new validation output directory")
    a.out.mkdir(parents=True)
    (a.out / "episodes").mkdir()
    tasks = json.loads((a.benchmark / "tasks.json").read_text())
    if a.split != "all":
        tasks = [t for t in tasks if t["split"] == a.split]
    if a.per_template:
        counts, selected = Counter(), []
        for t in tasks:
            if counts[t["template"]] < a.per_template:
                selected.append(t)
                counts[t["template"]] += 1
        tasks = selected
    begin, summaries = time.perf_counter(), []
    config = {"task_manifest_hash": digest(json.loads((a.benchmark / "tasks.json").read_text())),
              "core_hashes": core_fingerprint(),
              "selected_task_ids": [t["id"] for t in tasks], "backend": "owned real HAPI",
              "inference_api": False, "model_inference": False}
    (a.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    lock = threading.Lock()
    def worker(index):
      with HapiProcess(a.image, a.out / f"hapi-{index}", heap="3g") as store:
        for task in tasks[index::a.workers]:
            fixture = load_fixture(a.benchmark, task)
            assert digest(fixture) == task["fixture_hash"]
            before = store.reload(fixture)
            for hazard in task["hazards"]:
                conditions = ["vanilla", "medguard"] if hazard == "clean" else ["medguard"]
                for condition in conditions:
                    before = store.reset()
                    proxy = Proxy(store, task, hazard)
                    error = None
                    try:
                        answer = solve(public_task(task), proxy.call, condition)
                    except (ValueError, RuntimeError, KeyError) as e:
                        answer, error = {}, str(e)
                    after = store.snapshot()
                    result = grade(task, before, after, answer)
                    key = f"{task['id']}--{hazard}--{condition}"
                    row = {"key": key, "task_id": task["id"], "template": task["template"], "hazard": hazard,
                           "condition": condition, "grade": result, "triggered": proxy.triggered, "error": error,
                           "answer": answer, "trace": proxy.trace, "fixture_hash": task["fixture_hash"],
                           "before": before, "after": after}
                    save_gzip(a.out / "episodes" / (key + ".json.gz"), row)
                    summary = {k: row[k] for k in ("key", "task_id", "template", "hazard", "condition", "triggered", "error")}
                    summary["success"] = result["success"]
                    summary["valid"] = result["success"] and (hazard == "clean" or proxy.triggered > 0)
                    with lock:
                        summaries.append(summary)
                    print(json.dumps(summary), flush=True)
            with lock:
                (a.out / "progress.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(worker, range(a.workers)))
    report = {"episodes": len(summaries), "tasks": len(tasks), "valid": sum(r["valid"] for r in summaries),
              "failures": [r for r in summaries if not r["valid"]], "seconds": time.perf_counter() - begin,
              "all_valid": all(r["valid"] for r in summaries), "model_inference": False}
    (a.out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
