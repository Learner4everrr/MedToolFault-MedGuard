"""Read completed common shards for a clearly labelled, non-final progress report.

Never modifies the frozen study, prompts, audits, or official completion markers.
Selection depends only on completion/audit status, never model accuracy.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from medtoolbench.storage import digest
from .analyze import summarize
from .build import save_gzip
from .job_state import write_json
from .runtime import core_fingerprint
from .status import status
from .study_report import checked_rows
from .supervisor import load_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Use a new interim report directory")
    root, out = a.study.resolve(), a.out.resolve()
    plan = load_plan(root)
    if core_fingerprint() != plan["core_hashes"]:
        raise RuntimeError("Frozen core changed")
    if any(hashlib.sha256(Path(__file__).with_name(n + ".py").read_bytes()).hexdigest() != h
           for n, h in plan["analysis_hashes"].items()):
        raise RuntimeError("Frozen analysis code changed")
    manifest = json.loads((Path(plan["benchmark"]) / "tasks.json").read_text())
    if digest(manifest) != plan["manifest_hash"]:
        raise RuntimeError("Source manifest changed")
    tasks, systems = {t["id"]: t for t in manifest}, {s["name"]: s for s in plan["systems"]}
    completed, available = {}, defaultdict(set)
    for job in plan["jobs"]:
        path = root / "jobs" / job["id"] / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        proof = result["attempts"][-1].get("audit")
        if result["state"] == "complete" and proof and proof["all_pass"]:
            completed[job["id"]] = result
            available[job["model"]].add(job["task_shard"])
    common = sorted(set.intersection(*(available[s] for s in systems)))
    if not common:
        raise RuntimeError("No fully audited common shards across every system")
    selected = [j for j in plan["jobs"] if j["task_shard"] in common]
    selected_ids = {t for j in selected for t in j["task_ids"]}
    cohort = [t for t in manifest if t["id"] in selected_ids]
    out.mkdir(parents=True)
    selection = {"status": "INTERIM_DESCRIPTIVE_NOT_FINAL", "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                 "captured_at_chicago": datetime.now(ZoneInfo("America/Chicago")).isoformat(), "plan_hash": plan["plan_hash"],
                 "selection_rule": "All and only shards complete and independently audited for EVERY frozen system at capture; no accuracy-based selection",
                 "common_shards": common, "job_ids": [j["id"] for j in selected], "task_ids": sorted(selected_ids),
                 "tasks": len(cohort), "patient_groups": len({t["group"] for t in cohort}),
                 "action_required_tasks": sum(bool(t["contract"]["required_creations"]) for t in cohort),
                 "templates": dict(Counter(t["template"] for t in cohort)),
                 "hazard_variants_per_system_condition": dict(Counter(h for t in cohort for h in t["hazards"])),
                 "expected_episodes": sum(j["expected_episodes"] for j in selected),
                 "bootstrap_draws": plan["statistics"]["patient_cluster_bootstrap_draws"],
                 "interpretation": "Incomplete task subset and repeated interim look; exploratory descriptive intervals, not confirmatory significance or early stopping",
                 "inference_api": False, "additional_model_inference": False}
    write_json(out / "selection.json", selection)
    write_json(out / "live_status_at_capture.json", status(root.parent))
    print(json.dumps({k: v for k, v in selection.items() if k not in {"job_ids", "task_ids"}}), flush=True)
    def read_job(job):
        result = completed[job["id"]]
        run = Path(result["run"])
        config = json.loads((run / "config.json").read_text())
        rows, checksum = checked_rows(run, config, plan, job, systems[job["model"]], tasks)
        for row in rows:
            row["grade"] = {k: row["grade"][k] for k in ("success", "unsafe")}
        print(json.dumps({"loaded_job": job["id"], "episodes": len(rows)}), flush=True)
        return job, rows, {"run": str(run), "episodes_sha256": checksum, "episodes": len(rows)}
    by_model, provenance = defaultdict(list), {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for job, rows, proof in pool.map(read_job, selected):
            by_model[job["model"]].extend(rows)
            provenance[job["id"]] = proof
    reports = {}
    for system in plan["systems"]:
        name, rows = system["name"], by_model[system["name"]]
        keys = [(r["task_id"], r["hazard"], r["condition"]) for r in rows]
        if len(set(keys)) != len(keys) or set(r["task_id"] for r in rows) != selected_ids:
            raise RuntimeError("Nonidentical or duplicate model cohort")
        save_gzip(out / (name + "_analysis_rows.json.gz"), rows)
        reports[name] = {"family": system["family"], **summarize(rows,
            repetitions=selection["bootstrap_draws"], seed=plan["generation"]["seed"])}
        print(json.dumps({"summarized_model": name}), flush=True)
    metrics = {"selection": selection, "models": reports, "provenance": provenance}
    write_json(out / "metrics.json", metrics)
    with (out / "conditions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["system", "condition", "clean_success", "clean_n", "fault_success", "fault_n",
                         "conditional_success", "conditional_n", "unsafe", "unsafe_n", "timeout_duplicates", "timeout_n",
                         "mean_fault_tokens", "mean_fault_tool_calls"])
        for name, report in reports.items():
            for condition, v in report["conditions"].items():
                writer.writerow([name, condition, *[x for k in ("clean_success", "fault_success", "fault_success_clean_capable_baseline",
                    "persistent_unsafe", "commit_timeout_duplicates") for x in (v[k]["count"], v[k]["n"])],
                    v["mean_fault_tokens"], v["mean_fault_tool_calls"]])
    write_json(out / "complete.json", {"report_complete": True, "primary_study_complete": False,
                "episodes": sum(r["episodes"] for r in reports.values()), "systems": len(reports),
                "metrics_sha256": hashlib.sha256((out / "metrics.json").read_bytes()).hexdigest()})
    print(json.dumps({"interim_report": str(out), "episodes": selection["expected_episodes"]}), flush=True)


if __name__ == "__main__":
    main()
