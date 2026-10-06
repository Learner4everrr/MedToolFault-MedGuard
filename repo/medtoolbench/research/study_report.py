"""Complete-matrix reporting: no cherry-picked shards, patient-cluster intervals."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from medtoolbench.storage import digest
from .analyze import summarize
from .build import load_gzip
from .job_state import write_json

FIELDS = ("task_id", "hazard", "condition", "group", "template", "grade", "action_required",
          "any_transient_unsafe", "triggered", "stop", "total_tokens", "tool_calls")


def checked_rows(run, config, plan, job, system, tasks):
    if config["split"] != "test" or config["manifest_hash"] != plan["manifest_hash"]:
        raise RuntimeError("Non-test or wrong-manifest run in the primary study")
    if config.get("core_hashes") != plan["core_hashes"] or Path(config["model"]).resolve() != Path(system["path"]).resolve():
        raise RuntimeError("Source/model does not match the frozen plan")
    if config["thinking"] != system["thinking"] or config["batch_size"] != system["batch_size"]:
        raise RuntimeError("Model generation profile changed")
    if any(config.get(k) != v for k, v in plan["generation"].items()):
        raise RuntimeError("Generation limits changed")
    expected = {(task, hazard, condition) for task in job["task_ids"]
                for hazard in tasks[task]["hazards"] for condition in plan["conditions"]}
    seen, rows = set(), []
    content = hashlib.sha256()
    for path in sorted((run / "episodes").glob("*.json.gz")):
        # Hash the exact accepted episode bytes as a report provenance manifest.
        content.update(path.name.encode())
        content.update(hashlib.sha256(path.read_bytes()).digest())
        row = load_gzip(path)
        key = (row["task_id"], row["hazard"], row["condition"])
        if key not in expected or key in seen:
            raise RuntimeError("Unexpected or duplicate episode in completed shard")
        task = tasks[row["task_id"]]
        if task["split"] != "test" or row["group"] != task["group"] or row["template"] != task["template"]:
            raise RuntimeError("Episode patient group/template disagrees with the manifest")
        seen.add(key)
        rows.append({k: row[k] for k in FIELDS})
    if seen != expected or len(rows) != job["expected_episodes"]:
        raise RuntimeError("Missing episodes; partial matrices cannot issue a primary result")
    return rows, content.hexdigest()


def generate(root, plan):
    """Called by the last supervisor. Advisory lock makes a simultaneous finish safe."""
    actual_hashes = {name: hashlib.sha256(Path(__file__).with_name(name + ".py").read_bytes()).hexdigest()
                     for name in plan["analysis_hashes"]}
    if actual_hashes != plan["analysis_hashes"]:
        raise RuntimeError("Analysis/audit source changed after the study freeze")
    lockpath = Path("/tmp") / ("medtoolbench-report-" + digest(str(root.resolve()))[:16] + ".lock")
    with lockpath.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        output = root / "report"
        if (output / "complete.json").exists():
            previous = json.loads((output / "complete.json").read_text())
            if previous["plan_hash"] != plan["plan_hash"]:
                raise RuntimeError("Existing report belongs to another plan")
            return previous
        output.mkdir(exist_ok=True)
        source_tasks = json.loads((Path(plan["benchmark"]) / "tasks.json").read_text())
        if digest(source_tasks) != plan["manifest_hash"]:
            raise RuntimeError("Benchmark manifest changed after freezing")
        tasks = {t["id"]: t for t in source_tasks}
        summaries, provenance, count = {}, {}, 0
        for system in plan["systems"]:
            rows = []
            for job in (j for j in plan["jobs"] if j["model"] == system["name"]):
                result = json.loads((root / "jobs" / job["id"] / "result.json").read_text())
                proof = result["attempts"][-1].get("audit")
                if result["state"] != "complete" or not proof or not proof["all_pass"]:
                    raise RuntimeError("A shard has no passing independent audit")
                run = Path(result["run"])
                config = json.loads((run / "config.json").read_text())
                selected, checksum = checked_rows(run, config, plan, job, system, tasks)
                rows.extend(selected)
                provenance[job["id"]] = {"run": str(run), "episodes_sha256": checksum, "episodes": len(selected)}
            keys = [(r["task_id"], r["hazard"], r["condition"]) for r in rows]
            if len(keys) != len(set(keys)):
                raise RuntimeError("Repeated task/hazard/condition across shards")
            summaries[system["name"]] = {"family": system["family"], **summarize(rows,
                repetitions=plan["statistics"]["patient_cluster_bootstrap_draws"], seed=plan["generation"]["seed"])}
            count += len(rows)
            write_json(output / "progress.json", {"models_summarized": len(summaries), "episodes": count})
        if count != plan["expected_episodes"]:
            raise RuntimeError("Report matrix is incomplete")
        metrics = {"plan_hash": plan["plan_hash"], "split": "test", "episodes": count, "models": summaries,
                   "primary_estimand": plan["primary_estimand"], "statistics": plan["statistics"],
                   "provenance": provenance, "whole_research_project_complete": False}
        write_json(output / "metrics.json", metrics)
        lines = ["# Primary held-out results", "", "All frozen shards are included; no accuracy-based model or task exclusions.", "",
                 "| System | Clean-capable tasks | Paired guard gain | Patient-cluster 95% interval |",
                 "| --- | ---: | ---: | --- |"]
        def pct(value):
            return "undefined" if value is None else f"{value * 100:.2f} pp"
        for name, summary in summaries.items():
            effect = summary["paired_guard_gain_clean_capable"]
            interval = effect["patient_cluster_ci95"]
            bounds = "undefined" if interval is None else " to ".join(map(pct, interval))
            lines.append(f"| {name} | {effect['tasks']} | {pct(effect['task_weighted_gain'])} | {bounds} |")
        lines += ["", "Hazards are averaged within tasks, then patient groups are resampled 10,000 times.",
                  "Clean-capable eligibility is defined by each model's vanilla-clean outcome, identically for every intervention.",
                  "Untriggered fault assignments remain in their denominators. No confirmatory p-values or deployment-safety claim.",
                  "External evaluation, stress/ablation arms, and manuscript readiness are tracked separately.", ""]
        (output / "RESULTS.md").write_text("\n".join(lines))
        complete = {"plan_hash": plan["plan_hash"], "episodes": count, "systems": len(summaries),
                    "metrics_sha256": hashlib.sha256((output / "metrics.json").read_bytes()).hexdigest()}
        write_json(output / "complete.json", complete)
        return complete


if __name__ == "__main__":
    from .supervisor import load_plan
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(generate(a.study, load_plan(a.study)), indent=2))
