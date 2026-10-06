"""Regrade all saved outcomes independently and check stored DB/paired invariants."""
import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from .grading import grade
from .storage import digest


def audit(manifest, run):
    tasks = {t["id"]: t for t in json.loads(manifest.read_text())}
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    counts = Counter()
    cases = []
    for row in rows:
        task = tasks[row["task_id"]]
        initial = sorted(task["resources"], key=lambda r: (r["resourceType"], r["id"]))
        assert row["before"] == initial, (row["key"], "reset mismatch")
        again = grade(task, row["before"], row["after"], row["answer"])
        assert again == row["grade"], (row["key"], "grade mismatch")
        assert sum(x["injected"] for x in row["trace"]) == row["triggered"] <= 1
        if row["hazard"] == "clean":
            assert row["triggered"] == 0
        db_file = run / "states" / (row["key"] + ".sqlite")
        db = sqlite3.connect(db_file.resolve().as_uri() + "?mode=ro", uri=True)
        state = [json.loads(x[0]) for x in db.execute("SELECT body FROM resources ORDER BY kind,id")]
        db.close()
        assert state == row["after"], (row["key"], "persisted DB mismatch")
        assert row["total_tokens"] == sum(g["input_tokens"] + g["output_tokens"] for g in row["generations"])
        assert row["messages"][1]["content"] == task["prompt"]
        counts["episodes"] += 1
        counts["success"] += row["grade"]["success"]
        counts["hazard_untriggered"] += row["hazard"] != "clean" and not row["triggered"]
        counts["unsafe"] += bool(row["grade"]["unsafe"])
        if row["grade"]["unsafe"]:
            cases.append({"key": row["key"], "unsafe": row["grade"]["unsafe"],
                "hazard_triggered": row["triggered"], "answer": row["answer"],
                "diff": row["grade"]["diff"], "calls": [{"method": x["method"], "path": x["path"],
                "status": x["response"]["status"], "injected": x["injected"], "actor": x["actor"]} for x in row["trace"]]})
    return {"checks": dict(counts), "all_checks_passed": True, "unsafe_cases": cases}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    args = p.parse_args()
    for run in args.runs:
        if not (run / "complete.json").exists():
            raise SystemExit(f"Incomplete run: {run}")
        result = audit(args.manifest, run)
        (run / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(str(run), json.dumps(result["checks"]))


if __name__ == "__main__":
    main()
