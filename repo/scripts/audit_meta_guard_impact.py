"""Read every accepted action in completed evidence; preserve all old results."""
import argparse
import json
import time
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.build import load_gzip
from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.input_guard import validate_meta


def audit(roots, out):
    if out.exists():
        raise RuntimeError("Never overwrite an impact audit")
    out.mkdir(parents=True)
    summary, affected, references = [], [], []
    for root in roots:
        count, actions, hits = 0, 0, 0
        for path in sorted((root / "jobs").glob("*/result.json")):
            result = json.loads(path.read_text())
            if result.get("state") != "complete":
                continue
            run = Path(result["run"])
            proof = json.loads((run / "audit.json").read_text())
            if not proof["all_pass"]:
                raise RuntimeError("Completed evidence has a failed audit")
            references.append({"result": str(path), "result_hash": digest(result), "audit_hash": digest(proof)})
            seen = 0
            for file in sorted((run / "episodes").glob("*.json.gz")):
                row = load_gzip(file)
                count += 1
                seen += 1
                # This is the exact list accepted by the frozen parser, not a
                # regex match inside reasoning or an unexecuted JSON example.
                for index, action in enumerate(row.get("parsed_actions", [])):
                    actions += 1
                    try:
                        validate_meta(action)
                    except ValueError:
                        hits += 1
                        affected.append({"root": str(root), "job": path.parent.name,
                            "run": str(run), "episode": row["key"], "action_index": index,
                            "condition": row["condition"], "stage": row["split"],
                            "body_hash": digest(action.get("body")),
                            "meta_type": type(action["body"]["meta"]).__name__})
            if seen != result["episodes"] or seen != proof["episodes"]:
                raise RuntimeError("Completed evidence count mismatch: " + str(run))
            write_json(out / "progress.json", {"root": str(root), "episodes": count,
                "actions": actions, "affected_actions": hits, "heartbeat": time.time()})
        summary.append({"root": str(root), "episodes": count, "accepted_actions": actions, "affected_actions": hits})
    record = {"all_scanned": True, "behavior_unchanged_for_saved_trajectories": not affected,
        "affected": affected, "summary": summary, "references": references,
        "scope": "Completed primary and canonical supplemental results, including their qualification episodes; failed partial attempts are not pooled.",
        "historical_results_modified": False, "time": time.time()}
    write_json(out / "complete.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in {"references", "affected"}}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--roots", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    audit([r.resolve() for r in a.roots], a.out.resolve())
