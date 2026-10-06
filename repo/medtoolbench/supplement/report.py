"""Complete, audited follow-up tables with parent-task/patient-cluster effects."""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from medtoolbench.storage import digest
from medtoolbench.research.build import load_gzip
from medtoolbench.research.job_state import write_json
from .runner import load_plan


def paired(rows, treatment, metric="success", reference="vanilla", draws=10000):
    pairs, groups, parents = defaultdict(lambda: defaultdict(list)), {}, {}
    for r in rows:
        if r["hazard"] == "clean" or r["condition"] not in {reference, treatment}:
            continue
        key = (r["task"], r["hazard"])
        pairs[key][r["condition"]].append(float(r[metric]))
        groups[r["parent"]] = r["group"]
        parents[r["task"]] = r["parent"]
    parent_values = defaultdict(list)
    for (task, _), values in pairs.items():
        if set(values) == {reference, treatment}:
            parent_values[parents[task]].append(mean(values[treatment]) - mean(values[reference]))
    clusters = defaultdict(list)
    for parent, values in parent_values.items():
        clusters[groups[parent]].append(mean(values))
    if not clusters:
        return {"difference": None, "ci95": None, "parents": 0, "groups": 0}
    keys, rng = sorted(clusters), random.Random(20260917)
    samples = sorted(mean(v for _ in keys for v in clusters[rng.choice(keys)]) for _ in range(draws))
    return {"difference": mean(mean(v) for v in parent_values.values()),
        "ci95": [samples[int(.025 * draws)], samples[min(draws - 1, int(.975 * draws))]],
        "parents": len(parent_values), "groups": len(clusters), "bootstrap_draws": draws,
        "treatment": treatment, "reference": reference, "endpoint": metric}


def rate(rows, field):
    return {"n": len(rows), "count": sum(bool(r[field]) for r in rows), "rate": mean(bool(r[field]) for r in rows) if rows else None}


def generate(path):
    plan = load_plan(path)
    root, output = path.parent, path.parent / "report"
    if output.exists():
        if (output / "complete.json").exists():
            return
        raise RuntimeError("Unresolved partial report; preserve it and investigate")
    buckets, references = defaultdict(list), []
    for job in plan["jobs"]:
        result_path = root / "jobs" / job["id"] / "result.json"
        result = json.loads(result_path.read_text())
        if result["state"] != "complete":
            raise RuntimeError("Full-matrix report requires every shard")
        run = Path(result["run"])
        proof = json.loads((run / "audit.json").read_text())
        if not proof["all_pass"] or proof["episodes"] != len(job["episodes"]):
            raise RuntimeError("Invalid shard audit")
        wanted, seen = {e["key"] for e in job["episodes"]}, set()
        for file in sorted((run / "episodes").glob("*.json.gz")):
            r = load_gzip(file)
            if r["key"] in seen:
                raise RuntimeError("Duplicate episode")
            seen.add(r["key"])
            if job["stage"] == "qualification":
                continue
            external = "question_id" in r
            task = r["question_id"] if external else r["task_id"]
            row = {"task": task, "parent": r.get("parent_task", task), "group": r["group"],
                "template": r["template"], "hazard": r["hazard"], "condition": r["condition"],
                "success": r["success"] if external else r["grade"]["success"],
                "unsafe": False if external else bool(r["grade"]["unsafe"]),
                "duplicate": False if external else "duplicate" in r["grade"]["unsafe"],
                "unnecessary": False if external else "unnecessary_action" in r["grade"]["unsafe"],
                "transient_unsafe": r.get("any_transient_unsafe", False), "action_required": r.get("action_required", False),
                "triggered": bool(r["triggered"]), "tokens": r["input_tokens"] + r["output_tokens"],
                "calls": r.get("raw_tool_calls", r.get("tool_calls")), "code_calls": r.get("code_calls", 0),
                "external_writer": bool(r.get("external_writer_ids")), "stop": r["stop"], "read_only": external}
            buckets[(job["arm"], job["model"])].append(row)
        if seen != wanted:
            raise RuntimeError("Shard key mismatch")
        references.append({"job": job["id"], "run": str(run), "audit_hash": digest(proof)})
    unique_test_episodes = sum(len(v) for v in buckets.values())
    for (arm, model), rows in list(buckets.items()):
        if arm.startswith("sampling_"):
            buckets[("sampling_pooled", model)].extend(rows)
    reports = {}
    lines = ["# Supplemental results", "", "Post-primary follow-ups. Development checks are excluded. All assigned faults remain in denominators.",
        "Semi-synthetic task counts are not independent patients or authentic treatment decisions. External QA does not test write safety.", "",
        "| Arm | Model | Condition | Clean correct | Fault correct | Unsafe state (action-required) | Mean fault tokens |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for (arm, model), rows in sorted(buckets.items()):
        report = {"episodes": len(rows), "groups": len({r["group"] for r in rows}), "parents": len({r["parent"] for r in rows}),
            "conditions": {}, "comparisons": {}, "unsafe_comparisons": {}, "conditional_comparisons": {}}
        clean_capable = {r["task"] for r in rows if r["condition"] == "vanilla" and r["hazard"] == "clean" and r["success"]}
        report["vanilla_clean_capable_tasks"] = len(clean_capable)
        for condition in sorted({r["condition"] for r in rows}):
            chosen = [r for r in rows if r["condition"] == condition]
            clean, faults = [r for r in chosen if r["hazard"] == "clean"], [r for r in chosen if r["hazard"] != "clean"]
            actions = [r for r in faults if r["action_required"]]
            metrics = {"clean_success": rate(clean, "success"), "fault_success": rate(faults, "success"),
                "unsafe_state": None if all(r["read_only"] for r in rows) else rate(faults, "unsafe"),
                "action_required_unsafe_state": None if all(r["read_only"] for r in rows) else rate(actions, "unsafe"),
                "no_action_unnecessary": rate([r for r in faults if not r["action_required"] and not r["read_only"]], "unnecessary"),
                "duplicates": rate([r for r in faults if "commit_timeout" in r["hazard"]], "duplicate"),
                "transient_unsafe": rate([r for r in faults if not r["read_only"]], "transient_unsafe"),
                "triggered": rate(faults, "triggered"), "mean_fault_tokens": mean(r["tokens"] for r in faults) if faults else None,
                "mean_clean_tokens": mean(r["tokens"] for r in clean) if clean else None,
                "mean_fault_raw_calls": mean(r["calls"] for r in faults) if faults else None,
                "mean_code_calls": mean(r["code_calls"] for r in chosen), "stop_reasons": dict(Counter(r["stop"] for r in chosen)),
                "by_hazard": {h: rate([r for r in chosen if r["hazard"] == h], "success") for h in sorted({r["hazard"] for r in chosen})},
                "by_template": {str(t): rate([r for r in faults if r["template"] == t], "success") for t in sorted({r["template"] for r in chosen}, key=str)}}
            report["conditions"][condition] = metrics
            if condition != "vanilla":
                report["comparisons"][condition + "-vanilla"] = paired(rows, condition)
                # A pooled stochastic arm has no unique single-run eligibility;
                # do not define capability using "any seed happened to pass".
                if clean_capable and arm != "sampling_pooled":
                    report["conditional_comparisons"][condition + "-vanilla"] = paired([r for r in rows if r["task"] in clean_capable], condition)
                if actions:
                    report["unsafe_comparisons"][condition + "-vanilla"] = paired([r for r in rows if r["action_required"]], condition, "unsafe")
            if condition not in {"vanilla", "medguard"} and any(r["condition"] == "medguard" for r in rows):
                report["comparisons"][condition + "-medguard"] = paired(rows, condition, reference="medguard")
            def fmt(rate_value):
                return "not applicable" if rate_value is None or not rate_value["n"] else f"{rate_value['count']}/{rate_value['n']}"
            tokens = metrics["mean_fault_tokens"]
            lines.append(f"| {arm} | {model} | {condition} | {fmt(metrics['clean_success'])} | {fmt(metrics['fault_success'])} | {fmt(metrics['action_required_unsafe_state'])} | {round(tokens) if tokens is not None else 'NA'} |")
        if arm == "concurrent_writer":
            report["warning"] = "Unsafe state includes controlled external writes; this is not an agent-attributable harm rate. See external_writer_ids and traces."
        reports[arm + "/" + model] = report
    output.mkdir()
    write_json(output / "metrics.json", {"plan_hash": plan["plan_hash"], "reports": reports, "runs": references,
        "statistics": "10,000 patient-cluster percentile bootstrap draws; average faults/branches within parent task; no confirmatory p-values"})
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json(output / "complete.json", {"all_pass": True, "plan_hash": plan["plan_hash"], "episodes_test": unique_test_episodes,
        "metrics_hash": digest((output / "metrics.json").read_text())})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    generate(p.parse_args().plan.resolve())
