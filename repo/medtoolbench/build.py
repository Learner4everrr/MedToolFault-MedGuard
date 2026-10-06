import argparse
import json
import tempfile
from pathlib import Path

from .construction import VERSION, suite
from .grading import grade
from .hazards import Proxy
from .oracle import solve
from .storage import Store, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", type=int, default=40)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = suite(args.tasks)
    path = args.out / "tasks.json"
    if path.exists():
        raise SystemExit("Refusing to overwrite an existing manifest")
    results = []
    controls = []
    local_state = tempfile.TemporaryDirectory(prefix="medtoolbench-oracle-")
    for t in tasks:
        for hazard in t["hazards"]:
            s = Store(Path(local_state.name) / "oracle.sqlite", t["resources"])
            before = s.snapshot()
            proxy = Proxy(s, t, hazard)
            answer = solve(t, proxy)
            g = grade(t, before, s.snapshot(), answer)
            if not g["success"] or proxy.triggered != int(hazard != "clean"):
                raise RuntimeError((t["id"], hazard, g))
            results.append({"task_id": t["id"], "hazard": hazard, "answer": answer, "grade": g,
                            "trace": proxy.trace, "before": before, "after": s.snapshot()})
            s.reset()
            assert before == s.snapshot()
            s.close()
            for condition in ("vanilla", "retry_only"):
                s = Store(Path(local_state.name) / "oracle.sqlite", t["resources"])
                proxy = Proxy(s, t, hazard)
                answer = solve(t, proxy, condition)
                controls.append({"task_id": t["id"], "hazard": hazard, "condition": condition,
                    "grade": grade(t, before, s.snapshot(), answer), "trace": proxy.trace})
                s.close()
    path.write_text(json.dumps(tasks, indent=2) + "\n")
    (args.out / "oracle.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (args.out / "scripted_controls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in controls))
    meta = {"version": VERSION, "tasks": len(tasks), "episodes": len(results),
            "oracle_passed": len(results), "manifest_sha256": digest(tasks),
            "provenance": "Programmatic synthetic fixtures. Not original MAB-v3 data.",
            "patient_independent_tasks": len(tasks), "task_templates": 4,
            "action_required": sum(bool(t["contract"]["required_creations"]) for t in tasks),
            "no_action_conditional": sum(t["family"] == "conditional_medication" and not t["contract"]["required_creations"] for t in tasks)}
    (args.out / "build.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    local_state.cleanup()


if __name__ == "__main__":
    main()
