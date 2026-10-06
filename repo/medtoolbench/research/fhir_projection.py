"""Audit published lab/vital labels using a projection built only from raw FHIR.

The projection never reads the original CSV clinical tables, reference answers or
gold evidence IDs. Published SQL is used only by the offline label auditor, never
as an agent tool. This is not yet an online external-agent oracle validation.
"""
import argparse
import csv
import gzip
import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
import time
from collections import Counter
from pathlib import Path

from .external_inventory import parse_array_answer
from .sql_reference import authorizer, equivalent_rows

SUPPORTED = {"chartevents", "labevents"}


def identifier(resource, suffix):
    values = {x["value"] for x in resource.get("identifier", [])
              if x.get("system", "").endswith("/" + suffix) and "value" in x}
    if len(values) > 1:
        raise ValueError(f"Conflicting {suffix} identifiers for {resource.get('id')}")
    return int(next(iter(values))) if values else None


def local_clock(value):
    """Original MIMIC SQL uses local naive time, not UTC-normalized time."""
    if value is None:
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", value):
        raise ValueError("Unsupported source timestamp: " + value)
    return value[:19].replace("T", " ")


def reference(resource, field, kind):
    ref = resource.get(field, {}).get("reference")
    if ref is None:
        return None
    if not ref.startswith(kind + "/") or ref.count("/") != 1:
        raise ValueError("Unsupported source reference: " + ref)
    return ref.split("/")[1]


def project(db, read):
    """read(kind) yields ALL resources of that kind, in source order."""
    db.executescript("""
        CREATE TABLE admissions(subject_id INTEGER, hadm_id INTEGER PRIMARY KEY, admittime TEXT, dischtime TEXT);
        CREATE TABLE icustays(subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER PRIMARY KEY, intime TEXT, outtime TEXT);
        CREATE TABLE labevents(subject_id INTEGER, hadm_id INTEGER, itemid INTEGER, charttime TEXT, valuenum REAL, fhir_id TEXT UNIQUE);
        CREATE TABLE chartevents(subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER, itemid INTEGER, charttime TEXT, valuenum REAL, fhir_id TEXT UNIQUE);
        CREATE TABLE d_items(itemid INTEGER PRIMARY KEY, label TEXT, linksto TEXT);
        CREATE TABLE d_labitems(itemid INTEGER PRIMARY KEY, label TEXT);
    """)
    patients = {}
    for r in read("Patient"):
        number = identifier(r, "patient")
        if number is None:
            raise ValueError("Patient lacks source identifier")
        patients[r["id"]] = number
    encounters = list(read("Encounter"))
    hospitals, stays = {}, {}
    for r in encounters:
        hadm = identifier(r, "encounter-hosp")
        if hadm is not None:
            pid = reference(r, "subject", "Patient")
            hospitals[r["id"]] = (patients[pid], hadm)
            period = r.get("period", {})
            db.execute("INSERT INTO admissions VALUES(?,?,?,?)", (
                patients[pid], hadm, local_clock(period.get("start")), local_clock(period.get("end"))))
    for r in encounters:
        stay = identifier(r, "encounter-icu")
        if stay is not None:
            pid = reference(r, "subject", "Patient")
            subject, hadm = hospitals[reference(r, "partOf", "Encounter")]
            if subject != patients[pid]:
                raise ValueError("ICU/hospital patient disagreement")
            stays[r["id"]] = (subject, hadm, stay)
            period = r.get("period", {})
            db.execute("INSERT INTO icustays VALUES(?,?,?,?,?)", (
                subject, hadm, stay, local_clock(period.get("start")), local_clock(period.get("end"))))
    counts, labels = Counter(), {"chartevents": {}, "labevents": {}}
    for r in read("Observation"):
        profiles = r.get("meta", {}).get("profile", [])
        categories = [k for k in SUPPORTED if any(p.endswith("/mimic-observation-" + k) for p in profiles)]
        if not categories:
            counts["other_observations_not_projected"] += 1
            continue
        if len(categories) != 1:
            raise ValueError("Ambiguous observation profile")
        kind = categories[0]
        suffix = "/mimic-chartevents-d-items" if kind == "chartevents" else "/mimic-d-labitems"
        codes = {(int(c["code"]), c["display"]) for c in r.get("code", {}).get("coding", [])
                 if c.get("system", "").endswith(suffix)}
        if len(codes) != 1:
            raise ValueError(f"Ambiguous/absent source assay coding: {r['id']}")
        item, label = next(iter(codes))
        if item in labels[kind] and labels[kind][item] != label:
            raise ValueError(f"Inconsistent source label for {kind}/{item}")
        labels[kind][item] = label
        pid = patients[reference(r, "subject", "Patient")]
        encounter = reference(r, "encounter", "Encounter")
        value = r.get("valueQuantity", {}).get("value")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("Non-numeric quantity")
        clock = local_clock(r.get("effectiveDateTime"))
        if kind == "chartevents":
            subject, hadm, stay = stays[encounter]
            if subject != pid:
                raise ValueError("Observation/ICU patient disagreement")
            db.execute("INSERT INTO chartevents VALUES(?,?,?,?,?,?,?)", (pid, hadm, stay, item, clock, value, r["id"]))
        else:
            subject, hadm = hospitals[encounter] if encounter else (pid, None)
            if subject != pid:
                raise ValueError("Observation/hospital patient disagreement")
            db.execute("INSERT INTO labevents VALUES(?,?,?,?,?,?)", (pid, hadm, item, clock, value, r["id"]))
        counts[kind] += 1
        if (counts["chartevents"] + counts["labevents"]) % 100000 == 0:
            print(json.dumps({"observations_projected": counts["chartevents"] + counts["labevents"]}), flush=True)
    db.executemany("INSERT INTO d_items VALUES(?,?,?)", [(i, s, "chartevents") for i, s in labels["chartevents"].items()])
    db.executemany("INSERT INTO d_labitems VALUES(?,?)", list(labels["labevents"].items()))
    for table in ("admissions", "icustays", "chartevents", "labevents", "d_items", "d_labitems"):
        columns = {r[1] for r in db.execute("PRAGMA table_info(" + table + ")")}
        for column in ("subject_id", "hadm_id", "stay_id", "itemid", "label", "charttime"):
            if column in columns:
                db.execute(f"CREATE INDEX {table}_{column} ON {table}({column})")
    db.commit()
    return {**counts, "admissions": len(hospitals), "icustays": len(stays), "patients": len(patients),
            "d_items": len(labels["chartevents"]), "d_labitems": len(labels["labevents"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Use a new FHIR projection audit directory")
    a.out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="medtoolbench-fhir-projection-") as temp:
        # Avoid per-page random reads over the shared network filesystem.
        local_source = Path(temp) / "resources.sqlite"
        print(json.dumps({"state": "staging_read_only_index_on_local_disk"}), flush=True)
        shutil.copyfile(a.inventory / "resources.sqlite", local_source)
        source = sqlite3.connect(local_source.as_uri() + "?mode=ro", uri=True)
        def read(kind):
            for (blob,) in source.execute("SELECT body_gzip FROM resources WHERE kind=? ORDER BY rowid", (kind,)):
                yield json.loads(gzip.decompress(blob))
        path = Path(temp) / "projection.sqlite"
        db = sqlite3.connect(path)
        counts = project(db, read)
        source.close()
        print(json.dumps({"projected": counts}), flush=True)
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        db.set_authorizer(authorizer)
        results = []
        for row in csv.DictReader(a.questions.open()):
            if row["main_table_name"] not in SUPPORTED:
                continue
            start = time.monotonic()
            db.set_progress_handler(lambda: int(time.monotonic() - start > 10), 10000)
            result = {"question_id": row["question_id"], "main_table": row["main_table_name"], "template": row["template"]}
            try:
                actual = [list(r) for r in db.execute(row["sql_query"]).fetchall()]
                expected = parse_array_answer(row["true_answer"])
                result.update(match=equivalent_rows(actual, expected), actual=actual, expected=expected, error=None)
            except (sqlite3.Error, ValueError, SyntaxError) as error:
                result.update(match=False, error=str(error))
            results.append(result)
        db.close()
        shutil.copy2(path, a.out / "projection.sqlite")
    with a.questions.open("rb") as stream:
        questions_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"questions": len(results), "matched": sum(r["match"] for r in results),
              "mismatches": [r for r in results if not r["match"]], "projected_counts": counts,
              "templates": len({r["template"] for r in results}), "sqlite_integrity": integrity,
              "questions_sha256": questions_hash, "source_manifest": json.loads((a.inventory / "inventory.json").read_text()),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "projection_reads_gold_evidence_or_csv_tables": False, "model_inference": False,
              "status": "Offline FHIR-source label audit, not online agent evaluation",
              "timestamp_semantics": "Preserve original local wall-clock time for published SQL; do not convert to UTC",
              "dictionary_scope": "All observed FHIR assay codes/displays, not an external terminology database"}
    (a.out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.out / "all_questions.json").write_text(json.dumps(results) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"source_manifest", "mismatches"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
