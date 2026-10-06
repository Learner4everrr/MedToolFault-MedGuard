"""Authentic-source construction with deduplication and patient-group splits."""
import argparse
import copy
import gzip
import json
import os
import re
from collections import Counter
from pathlib import Path

from medtoolbench.storage import codes, digest, subject
from .contracts import LAB_CODES, NOW, READ_CODES, derive, observations, quantity


def save_gzip(path, value):
    # mtime=0 makes fixtures byte-reproducible, including the gzip container.
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    temporary.replace(path)


def load_gzip(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def source_identity(t):
    return {k: t.get(k, "") for k in ("instruction", "context", "eval_MRN")}


def measurable(resource, code):
    try:
        quantity(resource, code)
        return True
    except ValueError:
        return False


def remap(resources):
    """FHIR client-assigned numeric IDs need opaque, nonnumeric local identifiers."""
    mapping = {f"{r['resourceType']}/{r['id']}":
               f"{r['resourceType']}/r-{digest([r['resourceType'], r['id']])[:20]}"
               for r in resources if r["resourceType"] != "Patient"}
    def walk(x):
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, dict):
            return {k: mapping.get(v, v) if k == "reference" and isinstance(v, str) else walk(v) for k, v in x.items()}
        return x
    result = walk(copy.deepcopy(resources))
    for r in result:
        key = f"{r['resourceType']}/{r['id']}"
        if key in mapping:
            r["id"] = mapping[key].split("/")[1]
    return result, mapping


def parameters(t, template):
    if template == 1:
        m = re.search(r"name (.+?) and DOB of (\d{4}-\d{2}-\d{2})", t["instruction"])
        if m is None:
            raise ValueError("Unsupported patient-lookup instruction")
        return {"name": m[1], "birthDate": m[2]}
    if template == 3:
        m = re.search(r'"(\d+)/(\d+) mmHg"', t["instruction"])
        if m is None:
            raise ValueError("Unsupported BP instruction")
        return {"bp": [int(m[1]), int(m[2])]}
    if template == 8:
        m = re.search(r'free text of the referral, "(.+)"', t["instruction"])
        if m is None:
            raise ValueError("Unsupported referral instruction")
        return {"note": m[1]}
    return {}


ANSWER_FORMAT = {1: '{"mrn": "MRN or Patient not found"}', 2: '{"age": integer}',
                 4: '{"value": number}', 6: '{"value": number}', 7: '{"value": number}',
                 10: '{"value": number or -1, "date": "YYYY-MM-DD" or null}'}


def load_fixture(directory, task):
    resources = []
    for name in task["fixture_files"]:
        resources.extend(load_gzip(directory / name))
    resources.extend(copy.deepcopy(task.get("distractor_resources", [])))
    keys = [(r["resourceType"], r["id"]) for r in resources]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate fixture identifiers")
    return sorted(resources, key=lambda r: (r["resourceType"], r["id"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", default="medtoolbench-patient-split-20260911")
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Refusing to overwrite a constructed benchmark")
    a.out.mkdir(parents=True)
    (a.out / "fixtures").mkdir()
    patients = json.loads((a.export / "patients.json").read_text())
    save_gzip(a.out / "fixtures/patients.json.gz", patients)
    patient_rows, mapping, pool = {}, {}, []
    for f in sorted(a.export.glob("S*.json")):
        source = json.loads(f.read_text())
        # Explicit source subset: four assay histories and ALL existing orders.
        # No changes to the selected clinical values or dates.
        selected = [r for r in source if r["resourceType"] != "Observation" or LAB_CODES.intersection(codes(r))]
        rows, ids = remap(selected)
        patient_rows[f.stem] = rows
        mapping.update(ids)
        pool.extend(r for r in rows if r["resourceType"] == "Observation" and "valueQuantity" in r)
        save_gzip(a.out / f"fixtures/{f.stem}.json.gz", rows)
    (a.out / "source_id_map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    sources = [("v2", json.loads((a.export / "source_tasks.json").read_text())),
               ("v1", json.loads((a.upstream / "MedAgentBench/data/medagentbench/test_data_v1.json").read_text()))]
    seen, tasks, excluded, duplicates = set(), [], [], []
    for version, source_tasks in sources:
        for t in source_tasks:
            identity = digest(source_identity(t))
            if identity in seen:
                duplicates.append({"version": version, "id": t["id"], "source_identity": identity})
                continue
            seen.add(identity)
            template = int(t["id"].split("_")[0].removeprefix("task"))
            patient = t.get("eval_MRN")
            try:
                params = parameters(t, template)
                rows = patients + patient_rows.get(patient, [])
                if template != 1 and patient not in patient_rows:
                    raise ValueError("Patient source export missing")
                contract = derive(template, patient, rows, params)
                if template == 1 and [contract["answer"]["mrn"]] != t["sol"]:
                    raise ValueError("Existing patient-lookup label disagrees with source state")
                code = READ_CODES.get(template)
                relevant = observations(rows, patient, code, template in {4, 5, 6}) if code else []
                # Strict trigger eligibility: require two different measurable values
                # for stale responses and at least one for identity/provenance faults.
                stale_candidates = [r for r in relevant[1:] if measurable(r, code) and quantity(r, code) != quantity(relevant[0], code)]
                distractors = sorted([r for r in pool if code and code in codes(r)
                                      and subject(r) != f"Patient/{patient}" and measurable(r, code)], key=lambda r: digest([identity, r["id"]]))
                distractor = next((r for r in distractors if relevant and quantity(r, code) != quantity(relevant[0], code)), None)
                hazards = ["clean", "transient_503"]
                if code:
                    hazards.append("interface_drift")
                if stale_candidates:
                    hazards.append("stale")
                if relevant and distractor:
                    hazards.extend(["patient_mismatch", "provenance_conflict"])
                if contract["required_creations"]:
                    hazards.extend(["commit_timeout", "no_commit_timeout"])
                group = patient or "lookup-negative-" + digest(params)[:16]
                split = "dev" if int(digest([a.seed, group])[:8], 16) % 5 == 0 else "test"
                prompt = t["instruction"] + "\n" + t.get("context", "")
                prompt += "\nUse finalized results at or before the benchmark time; a one-year cutoff means the same calendar date/time in the preceding year."
                if template == 9:
                    prompt += " If no potassium measurement exists, do not create orders."
                prompt += "\nReturn final JSON with this shape: " + ANSWER_FORMAT.get(template, '{"completed": true}') + "."
                task = {"id": "mtb-" + identity[:16], "family": f"task{template}", "template": template,
                        "patient": patient, "group": group, "split": split, "read_code": code,
                        "parameters": params, "prompt": prompt, "contract": contract, "hazards": hazards,
                        "fixture_files": ["fixtures/patients.json.gz"] + ([f"fixtures/{patient}.json.gz"] if patient else []),
                        "distractor_resources": [distractor] if distractor else [],
                        "stale_resource": stale_candidates[0] if stale_candidates else None,
                        "source": {"version": version, "id": t["id"], "identity": identity},
                        "provenance": "MedAgentBench public task + official HAPI source record subset; explicit serialization adaptation"}
                task["fixture_hash"] = digest(load_fixture(a.out, task))
                tasks.append(task)
            except (ValueError, KeyError, StopIteration) as error:
                excluded.append({"version": version, "id": t["id"], "reason": str(error)})
    groups = {s: {t["group"] for t in tasks if t["split"] == s} for s in ("dev", "test")}
    assert not groups["dev"] & groups["test"]
    manifest = a.out / "tasks.json"
    manifest.write_text(json.dumps(tasks, indent=2) + "\n")
    report = {"manifest_hash": digest(tasks), "clock": NOW, "seed": a.seed,
              "raw_tasks": sum(len(x[1]) for x in sources), "exact_source_duplicates": len(duplicates),
              "constructed_tasks": len(tasks), "construction_exclusions": excluded,
              "tasks_by_template": dict(Counter(t["family"] for t in tasks)),
              "tasks_by_split": dict(Counter(t["split"] for t in tasks)),
              "groups_by_split": {s: len(g) for s, g in groups.items()},
              "hazard_counts": dict(Counter(h for t in tasks for h in t["hazards"])),
              "action_required": sum(bool(t["contract"]["required_creations"]) for t in tasks),
              "duplicates": duplicates, "status": "CONSTRUCTED, NOT YET HAPI ORACLE VALIDATED",
              "original_labels_available": "Patient lookup only; upstream refsol implementation absent from this public checkout",
              "fixture_scope": "All 695 source Patient resources; task-patient MG/GLU/K/A1C histories and all orders; one authentic distractor observation where eligible"}
    (a.out / "construction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "duplicates"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
