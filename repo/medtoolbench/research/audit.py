"""Independent replay of saved actual-state diffs and run completeness checks."""
import argparse
import copy
import json
from pathlib import Path

from medtoolbench.storage import digest
from .build import load_gzip
from .contracts import grade
from .fhir import normalized_state
from .protocol import messages


def apply_diff(before, diff):
    rows = {(r["resourceType"], r["id"]): copy.deepcopy(r) for r in before}
    if len(rows) != len(before):
        raise ValueError("Duplicate initial resource identity")
    for r in diff["removed"]:
        key = (r["resourceType"], r["id"])
        if rows.get(key) != r:
            raise ValueError("Removed resource does not match initial state")
        del rows[key]
    for r in diff["added"]:
        key = (r["resourceType"], r["id"])
        if key in rows:
            raise ValueError("Added resource already exists in initial state")
        rows[key] = copy.deepcopy(r)
    for change in diff["changed"]:
        r = change["after"]
        key = (r["resourceType"], r["id"])
        if key not in rows or rows[key] != change["before"]:
            raise ValueError("Changed resource does not match initial state")
        rows[key] = copy.deepcopy(r)
    return [rows[k] for k in sorted(rows)]


def transient_issues(row, task, before):
    issues = []
    expected = [i for i, t in enumerate(row["trace"])
                if t["method"] in {"POST", "PUT", "DELETE"} and
                (t["response"]["status"] < 400 or
                 t.get("committed_response") and t["committed_response"]["status"] < 400)]
    actual = [s["trace_index"] for s in row["transient_states"]]
    if actual != expected:
        issues.append("transient snapshots do not cover every successful actual mutation")
    for s in row["transient_states"]:
        saved = s["grade"]
        if saved["before_hash"] != digest(before):
            issues.append("transient initial-state hash mismatch")
        replayed = grade(task, before, apply_diff(before, saved["diff"]), {})
        for field in ("success", "state_ok", "answer_ok", "unsafe", "required_matches"):
            if replayed[field] != saved[field]:
                issues.append("transient regrade mismatch: " + field)
    if bool(row["any_transient_unsafe"]) != any(s["grade"]["unsafe"] for s in row["transient_states"]):
        issues.append("transient unsafe summary mismatch")
    return issues


def audit(run, benchmark):
    config = json.loads((run / "config.json").read_text())
    tasks = json.loads((benchmark / "tasks.json").read_text())
    if config["manifest_hash"] != digest(tasks):
        raise ValueError("Manifest changed since inference")
    by_id = {t["id"]: t for t in tasks}
    rows = [load_gzip(p) for p in sorted((run / "episodes").glob("*.json.gz"))]
    seen, errors = set(), []
    for r in rows:
        issues = []
        if r["key"] in seen:
            issues.append("duplicate episode key")
        seen.add(r["key"])
        initial = (run / r["before_ref"]).resolve()
        if not initial.is_relative_to(run.resolve()):
            raise ValueError("Initial-state reference leaves run directory")
        before = load_gzip(initial)
        if digest(before) != r["grade"]["before_hash"]:
            issues.append("initial-state hash mismatch")
        after = apply_diff(before, r["grade"]["diff"])
        new = grade(by_id[r["task_id"]], before, after, r["answer"])
        for field in ("success", "state_ok", "answer_ok", "unsafe", "required_matches"):
            if new[field] != r["grade"][field]:
                issues.append("regrade mismatch: " + field)
        if digest(normalized_state(after)) != r["after_semantic_hash"]:
            issues.append("actual end-state semantic hash mismatch")
        if sum(bool(t["injected"]) for t in r["trace"]) != r["triggered"]:
            issues.append("hazard trigger count mismatch")
        if r["messages"][:2] != messages(by_id[r["task_id"]], r["condition"], r["hazard"]):
            issues.append("public prompt mismatch")
        for field in ("input_tokens", "output_tokens"):
            if sum(g[field] for g in r["generations"]) != r[field]:
                issues.append(field + " accounting mismatch")
        if r["total_tokens"] != r["input_tokens"] + r["output_tokens"]:
            issues.append("total token accounting mismatch")
        if r["tool_calls"] != len(r["trace"]):
            issues.append("tool call accounting mismatch")
        if r["verification_or_retry_calls"] != sum(t["actor"] != "agent" for t in r["trace"]):
            issues.append("verification/retry accounting mismatch")
        issues.extend(transient_issues(r, by_id[r["task_id"]], before))
        if r["hazard"] == "commit_timeout":
            for t in r["trace"]:
                if t["injected"] and (t["response"]["status"] != 504 or not t["committed_response"] or t["committed_response"]["status"] >= 400):
                    issues.append("ambiguous write not backed by a successful actual server commit")
        if r["hazard"] == "no_commit_timeout":
            for t in r["trace"]:
                if t["injected"] and (t["response"]["status"] != 504 or t["committed_response"] is not None):
                    issues.append("no-commit control has an invalid injected outcome")
        if issues:
            errors.append({"key": r["key"], "issues": issues})
    expected = set(config["expected_keys"])
    report = {"episodes": len(rows), "expected": len(expected), "missing": sorted(expected - seen),
              "unexpected": sorted(seen - expected), "errors": errors,
              "complete_marker": (run / "complete.json").exists(),
              "audit_version": 2,
              "auditor_source_hash": digest(Path(__file__).read_text()),
              "all_pass": bool(rows) and not errors and seen == expected and (run / "complete.json").exists()}
    (run / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.run, a.benchmark)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_pass"] else 2)
