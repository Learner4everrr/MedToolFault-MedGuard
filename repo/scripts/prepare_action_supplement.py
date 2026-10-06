"""Explicitly semisynthetic decision and write-attribution controls."""
import argparse
import copy
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from medtoolbench.storage import codes, digest, subject
from medtoolbench.research.build import load_fixture, load_gzip, save_gzip
from medtoolbench.research.contracts import CLOCK, NOW, derive, instant, observations, service
from medtoolbench.research.job_state import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    primary = json.loads((root / "data/primary_study_v1/plan.json").read_text())
    original = json.loads((Path(primary["benchmark"]) / "tasks.json").read_text())
    if a.out.exists():
        raise RuntimeError("Never overwrite action-construction evidence")
    a.out.mkdir(parents=True)
    (a.out / "fixtures").mkdir()
    tasks, exclusions = [], []
    def save(task, resources):
        rows = sorted(resources, key=lambda r: (r["resourceType"], r["id"]))
        if len({(r["resourceType"], r["id"]) for r in rows}) != len(rows):
            raise RuntimeError("Duplicate synthetic fixture identity")
        path = a.out.resolve() / "fixtures" / (task["id"] + ".json.gz")
        save_gzip(path, rows)
        task.update(supplement_fixture=str(path), fixture_hash=digest(rows))
        tasks.append(task)
    for source in original:
        if source["template"] in {5, 9, 10}:
            for branch in (0, 1):
                t = copy.deepcopy(source)
                t.update(id="s-" + digest([source["id"], "branch", branch])[:20], parent_task=source["id"],
                    evidence_type="semisynthetic_primary_decision_branch", arm="decision_branches")
                rows = load_fixture(Path(primary["benchmark"]), source)
                code = t["read_code"]
                latest = CLOCK - timedelta(minutes=1) if t["template"] != 10 or branch else CLOCK.replace(year=2022) - timedelta(days=1)
                relevant = [r for r in rows if r["resourceType"] == "Observation" and subject(r) == "Patient/" + t["patient"] and code in codes(r)]
                if relevant:
                    newest = max(instant(r["effectiveDateTime"]) for r in relevant)
                    shift = latest - timedelta(hours=2) - newest
                    for r in relevant:
                        r["effectiveDateTime"] = (instant(r["effectiveDateTime"]) + shift).isoformat()
                value, unit = ({5: ((1.4, 2.1)[branch], "mg/dL"), 9: ((3.2, 3.8)[branch], "mmol/L"), 10: (7., "%")})[t["template"]]
                lab = {"resourceType": "Observation", "id": "r-" + digest([t["id"], "latest"])[0:20], "status": "final",
                    "subject": {"reference": "Patient/" + t["patient"]}, "code": {"coding": [{"code": code}]},
                    "effectiveDateTime": latest.isoformat(), "valueQuantity": {"value": value, "unit": unit}}
                old = copy.deepcopy(lab)
                old["id"] = "r-" + digest([t["id"], "older"])[0:20]
                old["effectiveDateTime"] = (latest - timedelta(days=400 if t["template"] == 10 else 0, hours=1)).isoformat()
                if t["template"] in {5, 9}:
                    old["valueQuantity"]["value"] = {5: (2.1, 1.4), 9: (3.8, 3.2)}[t["template"]][branch]
                rows.extend([lab, old])
                t["stale_resource"] = old
                t["contract"] = derive(t["template"], t["patient"], rows, t["parameters"])
                t["hazards"] = ["clean", "interface_drift", "transient_503", "stale", "provenance_conflict"]
                if t.get("distractor_resources"):
                    t["hazards"].append("patient_mismatch")
                if t["contract"]["required_creations"]:
                    t["hazards"].extend(["commit_timeout", "no_commit_timeout"])
                save(t, rows)
        if source["contract"]["required_creations"]:
            for variant in ("equivalent", "different", "cancelled"):
                t = copy.deepcopy(source)
                t.update(id="s-" + digest([source["id"], "prior", variant])[:20], parent_task=source["id"],
                    evidence_type="semisynthetic_preexisting_order", arm="preexisting_" + variant)
                t["prompt"] += "\nFor this benchmark instance, carry out this NEW request once. Any existing equivalent record represents a prior request and does not fulfill this new request. Preserve prior records."
                rows = load_fixture(Path(primary["benchmark"]), source)
                for i, required in enumerate(source["contract"]["required_creations"]):
                    previous = copy.deepcopy(required)
                    previous["id"] = "r-" + digest([t["id"], i])[:20]
                    if variant == "cancelled":
                        previous["status"] = {"Observation": "entered-in-error", "ServiceRequest": "revoked", "MedicationRequest": "cancelled"}[previous["resourceType"]]
                    elif variant == "different":
                        if previous["resourceType"] == "MedicationRequest":
                            previous["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]["value"] += 1
                        elif previous["resourceType"] == "Observation":
                            previous["component"][0]["valueQuantity"]["value"] += 5
                        else:
                            previous["code"]["coding"][0]["code"] += "-different"
                    rows.append(previous)
                t["hazards"] = ["clean", "commit_timeout", "no_commit_timeout"]
                save(t, rows)
    # Independent source patients, but operational rules and shifted clocks are
    # deliberately synthetic: no claim these are authentic treatment decisions.
    verified = root / "data/external_validation_all_v1"
    questions = load_gzip(verified / "public_questions.json.gz")
    splits = {q["patient_fhir_id"]: q["patient_group_split"] for q in questions}
    selected = {}
    for pid, split in sorted(splits.items()):
        rows = load_gzip(verified / "patients" / pid / "http_resources.json.gz")
        candidates = defaultdict(list)
        for r in rows:
            value = r.get("valueQuantity", {}).get("value")
            if r["resourceType"] == "Observation" and r.get("status") in {"final", "amended", "corrected"} and isinstance(value, (int, float)) and not isinstance(value, bool) and r.get("effectiveDateTime"):
                for c in r.get("code", {}).get("coding", []):
                    if c.get("system", "").endswith(("mimic-d-labitems", "mimic-chartevents-d-items")):
                        candidates[(c["system"], c["code"], c.get("display", c["code"]))].append(r)
        valid = []
        for key, items in candidates.items():
            if not 2 <= len(items) <= 3000 or len({r["valueQuantity"].get("unit") for r in items}) != 1:
                continue
            if len({r["valueQuantity"]["value"] for r in items}) < 2:
                continue
            latest = max(instant(r["effectiveDateTime"]) for r in items)
            if len({r["valueQuantity"]["value"] for r in items if instant(r["effectiveDateTime"]) == latest}) > 1:
                continue
            valid.append(key)
        if not valid:
            exclusions.append({"patient": pid, "reason": "No unambiguous varying numeric complete assay history <=3000 records"})
            continue
        key = min(valid, key=lambda k: digest(["external-action-assay", pid, k]))
        items = copy.deepcopy(candidates[key])
        shift = CLOCK - timedelta(hours=1) - max(instant(r["effectiveDateTime"]) for r in items)
        for r in items:
            r["effectiveDateTime"] = (instant(r["effectiveDateTime"]) + shift).isoformat()
        patient = next(r for r in rows if r["resourceType"] == "Patient" and r["id"] == pid)
        selected[pid] = (key, items, patient, shift.total_seconds())
    for pid, (key, items, patient, shift_seconds) in selected.items():
        ordered = sorted(items, key=lambda r: (instant(r["effectiveDateTime"]), r["id"]), reverse=True)
        latest = ordered[0]
        old = next((r for r in ordered[1:] if r["valueQuantity"]["value"] != latest["valueQuantity"]["value"]), None)
        if old is None:
            raise RuntimeError("Varying history lost its distinct old value")
        threshold = statistics.median(r["valueQuantity"]["value"] for r in items)
        other = next(((p, data) for p, data in sorted(selected.items()) if p != pid and data[0][:2] == key[:2]), None)
        for comparison in ("lt", "ge"):
            tid = "s-" + digest([pid, key, comparison])[:20]
            action = latest["valueQuantity"]["value"] < threshold if comparison == "lt" else latest["valueQuantity"]["value"] >= threshold
            t = {"id": tid, "parent_task": "external-action-" + pid, "group": pid, "patient": pid, "split": splits[pid],
                "template": 101, "family": "external_operational_rule", "read_code": key[1], "parameters": {"code": key[1], "system": key[0], "threshold": threshold, "comparison": comparison},
                "evidence_type": "semisynthetic_external_operational_rule", "arm": "external_actions",
                "source_clock_shift_seconds": shift_seconds, "source_assay": key, "stale_resource": old,
                "hazards": ["clean", "interface_drift", "transient_503", "stale", "provenance_conflict"],
                "contract": {"answer": {"completed": True}, "required_creations": [service(pid, key[1], key[0])] if action else []}}
            relation = "strictly less than" if comparison == "lt" else "greater than or equal to"
            t["prompt"] = (f"This is a synthetic operational-rule test, not medical advice. For patient {pid}, find the latest finalized {key[2]} observation with code {key[1]} (system {key[0]}) at or before {NOW}. "
                f"If its numeric value is {relation} {threshold}, create exactly one NEW ServiceRequest to repeat this assay using that SAME code and coding system. Otherwise create nothing. "
                'Use the prescribed ServiceRequest serialization with authoredOn at the frozen time; preserve existing records. Return {"completed":true}.')
            fixture = [copy.deepcopy(patient), *copy.deepcopy(items)]
            if other:
                donor, data = other
                distractor = copy.deepcopy(max(data[1], key=lambda r: instant(r["effectiveDateTime"])))
                t["distractor_resources"] = [distractor]
                fixture.extend([copy.deepcopy(data[2]), distractor])
                t["hazards"].append("patient_mismatch")
            if action:
                t["hazards"].extend(["commit_timeout", "no_commit_timeout"])
            save(t, fixture)
    write_json(a.out / "tasks.json", tasks)
    write_json(a.out / "construction.json", {"tasks": len(tasks), "arms": dict(Counter(t["arm"] for t in tasks)),
        "test_tasks": sum(t["split"] == "test" for t in tasks), "external_patients": len(selected), "exclusions": exclusions,
        "new_human_labels": False, "clinical_treatment_validation": False,
        "interpretation": "All new tasks are semi-synthetic; external source patients are not an independent clinical treatment study"})
    print((a.out / "construction.json").read_text())


if __name__ == "__main__":
    main()
