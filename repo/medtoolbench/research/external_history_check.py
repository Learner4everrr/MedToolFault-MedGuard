"""Largest development-history check for the external retrieval adaptation."""
import argparse
import json
import shutil
import sqlite3
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

from medtoolbench.storage import digest
from .build import load_gzip, save_gzip
from .external_store import import_records, patient_records, verified_roundtrip
from .fhir import HapiProcess
from .fhir_projection import SUPPORTED, project
from .job_state import write_json
from .sql_reference import authorizer, equivalent_rows


def main():
    p = argparse.ArgumentParser()
    for name in ("inventory", "image", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Use a new history-check directory")
    a.out.mkdir(parents=True)
    questions = load_gzip(a.inventory / "questions.json.gz")
    eligible = {q["patient_fhir_id"] for q in questions if q["patient_group_split"] == "dev" and q["main_table_name"] in SUPPORTED}
    try:
        with tempfile.TemporaryDirectory(prefix="medtoolbench-history-check-") as temp:
            local = Path(temp) / "source.sqlite"
            shutil.copyfile(a.inventory / "resources.sqlite", local)
            db = sqlite3.connect(local.as_uri() + "?mode=ro", uri=True)
            counts = dict(db.execute("SELECT patient,COUNT(*) FROM resources WHERE kind IN ('Encounter','Observation') GROUP BY patient"))
            patient = sorted(eligible, key=lambda i: (-counts.get(i, 0), i))[0]
            resources = patient_records(db, patient)
            db.close()
            selected = [q for q in questions if q["patient_fhir_id"] == patient and q["main_table_name"] in SUPPORTED]
            config = {"split": "dev", "patient": patient, "selection": "Largest full development Patient/Encounter/Observation history; ties lexical",
                      "question_ids": [q["question_id"] for q in selected], "source_resources_hash": digest(resources),
                      "source_counts": dict(Counter(r["resourceType"] for r in resources)), "gold_evidence_selection": False,
                      "retrieval": "HTTP _id batches over COMPLETE import receipts, never gold evidence IDs", "inference_api": False}
            write_json(a.out / "config.json", config)
            print(json.dumps(config), flush=True)
            with HapiProcess(a.image, a.out / "hapi", heap="4g") as store:
                store.types = ("Patient", "Encounter", "Observation")
                def progress(n):
                    write_json(a.out / "progress.json", {"resources_loaded": n, "resources": len(resources), "heartbeat": time.time()})
                receipts = import_records(store, resources, progress)
                write_json(a.out / "import_receipts.json", receipts)
                returned = verified_roundtrip(store, resources, receipts)
            save_gzip(a.out / "http_resources.json.gz", returned)
            projection = sqlite3.connect(":memory:")
            projected_counts = project(projection, lambda kind: (r for r in returned if r["resourceType"] == kind))
            projection.set_authorizer(authorizer)
            checks = []
            for q in selected:
                started = time.monotonic()
                projection.set_progress_handler(lambda: int(time.monotonic() - started > 10), 10000)
                actual = [list(r) for r in projection.execute(q["sql_query"]).fetchall()]
                checks.append({"question_id": q["question_id"], "match": equivalent_rows(actual, q["true_answer"]),
                               "actual": actual, "expected": q["true_answer"]})
            projection.close()
            write_json(a.out / "checks.json", checks)
            result = {"semantic_roundtrip_equal": True, "resources": len(returned), "projected_counts": projected_counts,
                      "questions": len(checks), "matched": sum(c["match"] for c in checks), "all_pass": all(c["match"] for c in checks),
                      "model_inference": False, "scope": "One largest development history; not full external model evaluation"}
            write_json(a.out / "complete.json", result)
            print(json.dumps(result), flush=True)
    except BaseException:
        write_json(a.out / "failed.json", {"error": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
