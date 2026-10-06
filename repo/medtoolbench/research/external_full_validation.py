"""Full external lab/vital oracle validation through fresh owned HAPI histories.

Every eligible question and its complete patient history is included. SQL/answers
are offline evaluator inputs only; public question artifacts use an explicit
allowlist that excludes both gold IDs and template-instantiation SQL fields.
"""
import argparse
import json
import shutil
import sqlite3
import tempfile
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medtoolbench.storage import digest
from .build import load_gzip, save_gzip
from .external_store import import_records, patient_records, verified_roundtrip
from .fhir import HapiProcess
from .fhir_projection import SUPPORTED, project
from .job_state import write_json
from .sql_reference import authorizer, equivalent_rows

PUBLIC_FIELDS = ("question_id", "question", "assumption", "patient_fhir_id", "patient_group_split", "template")


def main():
    p = argparse.ArgumentParser()
    for name in ("inventory", "image", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.out.exists() or not 1 <= a.workers <= 4:
        raise SystemExit("Use a new full-validation directory and 1–4 workers")
    a.out.mkdir(parents=True)
    questions = [q for q in load_gzip(a.inventory / "questions.json.gz") if q["main_table_name"] in SUPPORTED]
    patients = sorted({q["patient_fhir_id"] for q in questions})
    if len(questions) != 1125:
        raise RuntimeError("External eligible question count differs from the audited source")
    save_gzip(a.out / "public_questions.json.gz", [{k: q[k] for k in PUBLIC_FIELDS} for q in questions])
    config = {"questions": len(questions), "patients": patients, "question_manifest_hash": digest(questions),
              "templates": len({q["template"] for q in questions}), "split_counts": dict(Counter(q["patient_group_split"] for q in questions)),
              "source_hashes": {n: digest(Path(__file__).with_name(n + ".py").read_text()) for n in
                                ("external_full_validation", "external_store", "fhir_projection", "sql_reference", "fhir")},
              "workers": a.workers, "model_inference": False, "gold_evidence_selection": False,
              "history_scope": "Every Patient/Encounter/Observation record for every question patient", "inference_api": False}
    write_json(a.out / "config.json", config)
    print(json.dumps({k: v for k, v in config.items() if k != "source_hashes"}), flush=True)
    reports, lock, start = [], threading.Lock(), time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="medtoolbench-external-full-") as temp:
            local = Path(temp) / "index.sqlite"
            shutil.copyfile(a.inventory / "resources.sqlite", local)
            def worker(index):
                source = sqlite3.connect(local.as_uri() + "?mode=ro", uri=True)
                try:
                    for patient in patients[index::a.workers]:
                        folder = a.out / "patients" / patient
                        folder.mkdir(parents=True)
                        resources = patient_records(source, patient)
                        selected = [q for q in questions if q["patient_fhir_id"] == patient]
                        with HapiProcess(a.image, folder / "hapi", heap="4g") as store:
                            store.types = ("Patient", "Encounter", "Observation")
                            def progress(n):
                                write_json(a.out / f"worker{index}.json", {"patient": patient, "resources_loaded": n,
                                           "resources": len(resources), "state": "loading", "heartbeat": time.time()})
                            receipts = import_records(store, resources, progress)
                            returned = verified_roundtrip(store, resources, receipts)
                        save_gzip(folder / "http_resources.json.gz", returned)
                        write_json(folder / "import_receipts.json", receipts)
                        db = sqlite3.connect(":memory:")
                        try:
                            counts = project(db, lambda kind: (r for r in returned if r["resourceType"] == kind))
                            db.set_authorizer(authorizer)
                            checks = []
                            for q in selected:
                                started = time.monotonic()
                                db.set_progress_handler(lambda: int(time.monotonic() - started > 10), 10000)
                                actual = [list(r) for r in db.execute(q["sql_query"]).fetchall()]
                                checks.append({"question_id": q["question_id"], "match": equivalent_rows(actual, q["true_answer"]),
                                               "actual": actual, "expected": q["true_answer"]})
                        finally:
                            db.close()
                        write_json(folder / "checks.json", checks)
                        report = {"patient": patient, "split": selected[0]["patient_group_split"], "resources": len(returned),
                                  "source_resources_hash": digest(resources), "http_resources_hash": digest(returned),
                                  "semantic_roundtrip_equal": True, "projected_counts": counts,
                                  "questions": len(checks), "matched": sum(c["match"] for c in checks), "all_pass": all(c["match"] for c in checks)}
                        write_json(folder / "complete.json", report)
                        with lock:
                            reports.append(report)
                            write_json(a.out / "progress.json", {"patients_complete": len(reports), "patients": len(patients),
                                       "questions_complete": sum(r["questions"] for r in reports), "questions": len(questions),
                                       "matched": sum(r["matched"] for r in reports), "heartbeat": time.time()})
                        print(json.dumps({k: v for k, v in report.items() if k != "projected_counts"}), flush=True)
                        del resources, returned
                finally:
                    source.close()
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                list(pool.map(worker, range(a.workers)))
        complete = {"patients": len(reports), "questions": sum(r["questions"] for r in reports), "matched": sum(r["matched"] for r in reports),
                    "resources": sum(r["resources"] for r in reports), "all_pass": len(reports) == len(patients) and all(r["all_pass"] for r in reports),
                    "failed_patients": [r["patient"] for r in reports if not r["all_pass"]], "seconds": time.monotonic() - start,
                    "model_inference": False, "inference_api": False, "scope": "Complete external lab/vital real-HAPI transport and reference-answer audit, not model performance"}
        write_json(a.out / "complete.json", complete)
        print(json.dumps(complete), flush=True)
        if not complete["all_pass"]:
            raise RuntimeError("Full external reference validation has mismatches")
    except BaseException:
        write_json(a.out / "failed.json", {"error": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
