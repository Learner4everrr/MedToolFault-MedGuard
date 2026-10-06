"""Reproduce external published SQL labels on the author's public CSV substrate.

This is a label audit only. The model-facing external environment remains FHIR;
neither SQL queries nor ground-truth IDs are supplied to a model.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path

from .external_inventory import parse_array_answer

TABLES = ["patients", "admissions", "d_icd_diagnoses", "d_icd_procedures", "d_items", "d_labitems",
          "diagnoses_icd", "procedures_icd", "labevents", "prescriptions", "chartevents", "inputevents",
          "outputevents", "microbiologyevents", "icustays", "transfers"]


def equivalent_rows(actual, expected):
    if len(actual) != len(expected):
        return False
    def cell(a, b):
        if a is None or b is None:
            return a is b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
        return str(a).strip().casefold() == str(b).strip().casefold()
    remaining = list(expected)
    for row in actual:
        index = next((i for i, other in enumerate(remaining) if len(row) == len(other) and all(cell(a, b) for a, b in zip(row, other))), None)
        if index is None:
            return False
        remaining.pop(index)
    return True


def authorizer(action, arg1, arg2, database, trigger):
    if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ}:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in {
        "count", "sum", "min", "max", "avg", "datetime", "date", "time", "strftime", "julianday",
        "round", "abs", "lower", "upper", "trim", "ltrim", "rtrim", "coalesce", "ifnull", "like", "instr", "length", "substr"}:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def main():
    import pandas as pd
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Use a new SQL label-audit directory")
    a.out.mkdir(parents=True)
    with a.zip.open("rb") as f:
        archive_hash = hashlib.file_digest(f, "sha256").hexdigest()
    with tempfile.TemporaryDirectory(prefix="medtoolbench-sql-reference-") as temp, zipfile.ZipFile(a.zip) as archive:
        db_path = Path(temp) / "reference.sqlite"
        db = sqlite3.connect(db_path)
        names = archive.namelist()
        def read(table):
            paths = [n for n in names if n.endswith("/" + table + ".csv.gz")]
            if len(paths) != 1:
                raise ValueError(f"Ambiguous source table: {table}")
            with archive.open(paths[0]) as stream:
                return pd.read_csv(stream, compression="gzip", low_memory=False,
                                   dtype={"ndc": str, "icd_code": str})
        counts = {}
        for table in TABLES:
            frame = read(table)
            if table == "procedures_icd":
                frame["charttime"] = pd.to_datetime(frame["chartdate"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            elif table == "diagnoses_icd":
                frame = frame.merge(read("admissions")[["hadm_id", "admittime"]], on="hadm_id", how="left")
                frame.rename(columns={"admittime": "charttime"}, inplace=True)
            frame.insert(0, "row_id", [f"{table}_{i}" for i in range(len(frame))])
            frame.to_sql(table, db, index=False, chunksize=10000)
            for column in ("subject_id", "hadm_id", "stay_id", "itemid", "icd_code", "label", "drug", "charttime"):
                if column in frame.columns:
                    db.execute(f'CREATE INDEX "{table}_{column}" ON "{table}"("{column}")')
            counts[table] = len(frame)
            print(json.dumps({"table": table, "rows": len(frame)}), flush=True)
            del frame
        db.commit()
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        db.set_authorizer(authorizer)
        results = []
        for row in csv.DictReader(a.questions.open()):
            started = time.monotonic()
            db.set_progress_handler(lambda: int(time.monotonic() - started > 10), 10000)
            try:
                actual = [list(r) for r in db.execute(row["sql_query"]).fetchall()]
                expected = parse_array_answer(row["true_answer"])
                result = {"question_id": row["question_id"], "match": equivalent_rows(actual, expected),
                          "actual": actual, "expected": expected, "error": None}
            except (sqlite3.Error, ValueError, SyntaxError) as error:
                result = {"question_id": row["question_id"], "match": False, "error": str(error)}
            results.append(result)
        db.close()
        shutil.copy2(db_path, a.out / "reference.sqlite")
    report = {"questions": len(results), "matched": sum(r["match"] for r in results),
              "mismatches": [r for r in results if not r["match"]], "source_archive_sha256": archive_hash,
              "table_rows": counts, "sqlite_integrity": integrity, "model_inference": False,
              "status": "SQL-source label audit, not external FHIR oracle validation"}
    (a.out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.out / "all_questions.json").write_text(json.dumps(results) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
