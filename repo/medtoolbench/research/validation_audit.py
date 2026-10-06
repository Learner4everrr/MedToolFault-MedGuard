"""Regrade complete oracle artifacts and verify the declared validation matrix."""
import argparse
import json
from pathlib import Path

from medtoolbench.storage import digest
from .build import load_fixture, load_gzip
from .contracts import grade
from .fhir import normalized_state


def audit(validation, benchmark):
    config = json.loads((validation / "config.json").read_text())
    complete = json.loads((validation / "complete.json").read_text())
    manifest = json.loads((benchmark / "tasks.json").read_text())
    if digest(manifest) != config["task_manifest_hash"]:
        raise ValueError("Validation manifest changed")
    selected = set(config["selected_task_ids"])
    tasks = {t["id"]: t for t in manifest if t["id"] in selected}
    if set(tasks) != selected:
        raise ValueError("Validation selects unknown tasks")
    expected = {f"{t['id']}--{h}--{c}" for t in tasks.values() for h in t["hazards"]
                for c in (["vanilla", "medguard"] if h == "clean" else ["medguard"])}
    fixtures = {t["id"]: normalized_state(load_fixture(benchmark, t)) for t in tasks.values()}
    seen, errors, timeouts = set(), [], {}
    for path in sorted((validation / "episodes").glob("*.json.gz")):
        row = load_gzip(path)
        key, task = row["key"], tasks[row["task_id"]]
        problems = []
        if key in seen:
            problems.append("duplicate episode key")
        seen.add(key)
        new = grade(task, row["before"], row["after"], row["answer"])
        for field in ("success", "state_ok", "answer_ok", "unsafe", "required_matches", "before_hash", "after_hash"):
            if new[field] != row["grade"][field]:
                problems.append("regrade mismatch: " + field)
        if not new["success"] or row["error"] is not None:
            problems.append("oracle failed")
        if normalized_state(row["before"]) != fixtures[row["task_id"]]:
            problems.append("initial state is not the declared full fixture")
        injected = [t for t in row["trace"] if t["injected"]]
        if len(injected) != row["triggered"] or (row["hazard"] != "clean" and not injected):
            problems.append("injection count/eligibility failure")
        if row["hazard"] in {"commit_timeout", "no_commit_timeout"}:
            if len(injected) != 1 or injected[0]["response"]["status"] != 504:
                problems.append("invalid ambiguous outcome")
            else:
                t = injected[0]
                committed = t["committed_response"]
                if row["hazard"] == "commit_timeout" and (not committed or committed["status"] >= 400):
                    problems.append("commit control lacks successful actual commit")
                if row["hazard"] == "no_commit_timeout" and committed is not None:
                    problems.append("no-commit control performed a commit")
                timeouts[(row["task_id"], row["hazard"])] = t["response"]
        if problems:
            errors.append({"key": key, "problems": problems})
    pairs = 0
    for task in tasks.values():
        if "commit_timeout" in task["hazards"] and "no_commit_timeout" in task["hazards"]:
            pairs += 1
            committed = timeouts.get((task["id"], "commit_timeout"))
            uncommitted = timeouts.get((task["id"], "no_commit_timeout"))
            if committed is None or committed != uncommitted:
                errors.append({"task_id": task["id"], "problems": ["paired timeout public responses differ or are absent"]})
    report = {"episodes": len(seen), "expected": len(expected), "tasks": len(tasks),
              "missing": sorted(expected - seen), "unexpected": sorted(seen - expected), "errors": errors,
              "identical_timeout_pairs": pairs, "manifest_hash": digest(manifest),
              "auditor_source_hash": digest(Path(__file__).read_text()),
              "all_pass": bool(expected) and seen == expected and not errors and complete["all_valid"] and complete["episodes"] == len(expected)}
    (validation / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.validation, a.benchmark)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_pass"] else 2)
