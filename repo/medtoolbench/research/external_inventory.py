"""Inventory the complete public external substrate without a cloud service.

All resources are indexed, never only gold evidence. This is data preparation,
not yet validation of question semantics or external model performance.
"""
import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import shutil
import sqlite3
import tempfile
import tokenize
from collections import Counter
from pathlib import Path

from medtoolbench.storage import digest
from .build import save_gzip


def parse_array_answer(raw):
    """Parse published Python/NumPy array renderings without eval or code calls.

    NumPy's array string omits separators, including between quoted cells. Insert
    only token-boundary commas; never rewrite the content of a quoted string.
    """
    tokens, previous = [], None
    ignored = {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    for token in tokenize.generate_tokens(io.StringIO(raw).readline):
        if token.type in ignored:
            continue
        if token.type == tokenize.NAME and token.string not in {"None", "True", "False"}:
            raise ValueError("Unsupported answer literal: " + token.string)
        if token.type not in {tokenize.STRING, tokenize.NUMBER, tokenize.NAME, tokenize.OP}:
            raise ValueError("Unsupported answer token")
        if token.type == tokenize.OP and token.string not in {"[", "]", ",", "-", "+"}:
            raise ValueError("Answer contains non-literal syntax")
        left = previous is not None and (previous.type in {tokenize.STRING, tokenize.NUMBER, tokenize.NAME} or previous.string == "]")
        right = token.type in {tokenize.STRING, tokenize.NUMBER, tokenize.NAME} or token.string in {"[", "-", "+"}
        if left and right:
            tokens.append((tokenize.OP, ","))
        tokens.append((token.type, token.string))
        previous = token
    return ast.literal_eval(tokenize.untokenize(tokens))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Refusing to overwrite the external inventory")
    a.out.mkdir(parents=True)
    source_hashes = {}
    for line in (a.demo / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        with (a.demo / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Upstream checksum mismatch: {name}")
        source_hashes[name] = actual
    counts, patient_refs = Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix="medtoolbench-external-index-") as temp:
        database = Path(temp) / "resources.sqlite"
        db = sqlite3.connect(database)
        db.execute("CREATE TABLE resources(kind TEXT,id TEXT,patient TEXT,body_gzip BLOB,PRIMARY KEY(kind,id))")
        for path in sorted((a.demo / "fhir").glob("*.ndjson.gz")):
            with db, gzip.open(path, "rt") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    resource = json.loads(line)
                    kind = resource["resourceType"]
                    ref = resource.get("subject", resource.get("patient", {})).get("reference")
                    patient = ref.split("/")[-1] if ref else resource["id"] if kind == "Patient" else None
                    # Raw clinical record preserved; gzip only changes representation.
                    db.execute("INSERT INTO resources VALUES(?,?,?,?)", (kind, resource["id"], patient,
                               gzip.compress(json.dumps(resource, separators=(",", ":")).encode(), mtime=0)))
                    counts[kind] += 1
                    if patient:
                        patient_refs[patient] += 1
            print(json.dumps({"file": path.name, "resources_indexed": sum(counts.values())}), flush=True)
        db.execute("CREATE INDEX resources_patient_kind ON resources(patient,kind)")
        db.commit()
        rows = list(csv.DictReader(a.questions.open()))
        questions, missing = [], []
        for r in rows:
            gold_ids = ast.literal_eval(r["true_fhir_ids"])
            patient = r["patient_fhir_id"]
            absent = []
            for kind, ids in gold_ids.items():
                for rid in ids:
                    if not db.execute("SELECT 1 FROM resources WHERE kind=? AND id=?", (kind, rid)).fetchone():
                        absent.append([kind, rid])
            if absent:
                missing.append({"question_id": r["question_id"], "missing_gold_ids": absent})
            split = "dev" if int(digest(["external-patient-split-20260911", patient])[:8], 16) % 5 == 0 else "test"
            questions.append({**r, "raw_true_answer": r["true_answer"], "true_answer": parse_array_answer(r["true_answer"]), "true_fhir_ids": gold_ids,
                              "patient_group_split": split, "all_gold_resources_present": not absent,
                              "patient_resource_present": bool(db.execute("SELECT 1 FROM resources WHERE kind='Patient' AND id=?", (patient,)).fetchone())})
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        db.close()
        if integrity != "ok":
            raise RuntimeError(integrity)
        shutil.copy2(database, a.out / "resources.sqlite")
    save_gzip(a.out / "questions.json.gz", questions)
    report = {"status": "inventory only; question semantics not yet oracle-validated", "questions": len(questions),
              "patients_in_source": counts["Patient"], "resource_counts": dict(counts),
              "source_hashes": source_hashes, "questions_hash": digest(questions),
              "question_template_count": len({q["template"] for q in questions}),
              "questions_by_patient_group_split": dict(Counter(q["patient_group_split"] for q in questions)),
              "questions_missing_gold_resources": missing,
              "questions_missing_patient": [q["question_id"] for q in questions if not q["patient_resource_present"]],
              "sqlite_integrity": integrity, "inference_api": False, "cloud_service": False,
              "license": "Original MIMIC-IV-FHIR Demo: ODbL 1.0; author question dataset: CC BY 4.0. No redistribution performed."}
    (a.out / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"source_hashes", "questions_missing_gold_resources"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
