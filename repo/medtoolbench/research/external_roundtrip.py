"""Development-only local HAPI roundtrip and SQL-reference checks for external FHIR.

Patient selection uses a fixed patient split, never answer values or model outcomes.
All Patient/Encounter/Observation resources for those patients are transferred;
gold evidence IDs never select records. Only reference queries see SQL or answers.
"""
import argparse
import gzip
import json
import shutil
import sqlite3
import tempfile
import time
from collections import Counter
from pathlib import Path

from medtoolbench.storage import digest
from .build import load_gzip, save_gzip
from .fhir import HapiProcess, normalized_state
from .fhir_projection import SUPPORTED, project
from .sql_reference import authorizer, equivalent_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--patients", type=int, default=2)
    a = p.parse_args()
    if a.out.exists() or a.patients < 1:
        raise SystemExit("Require a new output directory and positive patient count")
    a.out.mkdir(parents=True)
    all_questions = load_gzip(a.inventory / "questions.json.gz")
    patients = sorted({q["patient_fhir_id"] for q in all_questions
                       if q["patient_group_split"] == "dev" and q["main_table_name"] in SUPPORTED})[:a.patients]
    questions = [q for q in all_questions if q["patient_fhir_id"] in patients and q["main_table_name"] in SUPPORTED]
    (a.out / "config.json").write_text(json.dumps({"patients": patients, "split": "dev", "question_ids": [q["question_id"] for q in questions],
        "selection": "First lexicographically sorted eligible development patient IDs", "gold_evidence_selection": False,
        "source_hash": digest(Path(__file__).read_text()), "inference_api": False}, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="medtoolbench-external-roundtrip-") as temp:
        index = Path(temp) / "resources.sqlite"
        shutil.copyfile(a.inventory / "resources.sqlite", index)
        db = sqlite3.connect(index.as_uri() + "?mode=ro", uri=True)
        resources = []
        for patient in patients:
            for (blob,) in db.execute("SELECT body_gzip FROM resources WHERE (kind='Patient' AND id=?) OR (patient=? AND kind IN ('Encounter','Observation')) ORDER BY rowid", (patient, patient)):
                resources.append(json.loads(gzip.decompress(blob)))
        db.close()
        # HAPI's reference search indexes must see target resources at insertion.
        # Disabling referential-integrity enforcement does not defer/rebuild those
        # indexes automatically when Patient resources arrive later.
        resources.sort(key=lambda r: (0 if r["resourceType"] == "Patient" else
                                      2 if r["resourceType"] == "Encounter" and r.get("partOf") else
                                      1 if r["resourceType"] == "Encounter" else 3, r["id"]))
        print(json.dumps({"selected_patients": len(patients), "questions": len(questions), "resources": len(resources)}), flush=True)
        with HapiProcess(a.image, a.out / "hapi", heap="3g") as store:
            store.types = ("Patient", "Encounter", "Observation")
            if store.snapshot():
                raise RuntimeError("New external HAPI sandbox is not empty")
            for offset in range(0, len(resources), 200):
                # Keep source profiles; HAPI supplies its own version/update metadata.
                batch = resources[offset:offset+200]
                entries = [{"request": {"method": "PUT", "url": r["resourceType"] + "/" + r["id"]}, "resource": r} for r in batch]
                store.transaction(entries)
                print(json.dumps({"resources_loaded": min(offset + 200, len(resources))}), flush=True)
            # Partition by patient: the upstream image caps unqualified searches
            # at 5,000 results. Equality against all source records still gates
            # acceptance, so any within-patient cap remains a visible failure.
            returned = []
            for patient in patients:
                response = store.call("GET", "/Patient/" + patient)
                if response["status"] != 200:
                    raise RuntimeError("Patient read failed after successful import")
                returned.append(response["body"])
                for kind in ("Encounter", "Observation"):
                    returned.extend(store.all_pages(f"/{kind}?subject=Patient/{patient}&_count=2000&_total=accurate"))
            expected_state, actual_state = normalized_state(resources), normalized_state(returned)
            roundtrip = expected_state == actual_state
            save_gzip(a.out / "http_returned_resources.json.gz", returned)
            save_gzip(a.out / "source_resources.json.gz", resources)
            if not roundtrip:
                old = {tuple(r["key"]): r for r in expected_state}
                new = {tuple(r["key"]): r for r in actual_state}
                differences = [{"key": list(k), "source": old.get(k), "returned": new.get(k)}
                               for k in sorted(old.keys() | new.keys()) if old.get(k) != new.get(k)]
                save_gzip(a.out / "roundtrip_differences.json.gz", differences)
                raise RuntimeError(f"FHIR roundtrip changed {len(differences)} source records; inspect retained artifacts")
        projected = sqlite3.connect(":memory:")
        counts = project(projected, lambda kind: (r for r in returned if r["resourceType"] == kind))
        projected.set_authorizer(authorizer)
        results = []
        for q in questions:
            started = time.monotonic()
            projected.set_progress_handler(lambda: int(time.monotonic() - started > 10), 10000)
            actual = [list(r) for r in projected.execute(q["sql_query"]).fetchall()]
            results.append({"question_id": q["question_id"], "match": equivalent_rows(actual, q["true_answer"]),
                            "actual": actual, "expected": q["true_answer"]})
        projected.close()
    report = {"split": "dev", "patients": len(patients), "resource_counts": dict(Counter(r["resourceType"] for r in resources)),
              "semantic_roundtrip_equal": roundtrip, "questions": len(results), "matched": sum(r["match"] for r in results),
              "mismatches": [r for r in results if not r["match"]], "projected_counts": counts,
              "all_pass": roundtrip and all(r["match"] for r in results), "inference_api": False,
              "retrieval": "Real HTTP, patient-partitioned paginated searches, complete-source equality gate",
              "status": "Development-only real HAPI transport/reference-query validation, not model performance"}
    (a.out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.out / "all_questions.json").write_text(json.dumps(results) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
