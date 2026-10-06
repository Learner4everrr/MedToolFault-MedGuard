"""Seeded synthetic engineering suite. NOT original MAB patient episodes."""
import copy
import random

NOW = "2023-11-13T10:15:00Z"
LATEST = "2023-11-13T09:15:00Z"
OLDER = "2023-11-13T05:15:00Z"
VERSION = "synthetic-mechanism-v0.1"


def concept(system, code):
    return {"coding": [{"system": system, "code": code}]}


def observation(pid, rid, code, value, unit, when):
    return {"resourceType": "Observation", "id": rid, "status": "final",
            "code": concept("http://loinc.org", code), "subject": {"reference": f"Patient/{pid}"},
            "effectiveDateTime": when, "valueQuantity": {"value": value, "unit": unit}}


def suite(n=40, seed=20260911):
    if n % 4:
        raise ValueError("n must be a multiple of four")
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        pid, other = f"P{10000+i}", f"P{20000+i}"
        family = ("latest_lab", "conditional_medication", "record_vital", "referral")[i % 4]
        base = [{"resourceType": "Patient", "id": p, "active": True,
                 "name": [{"family": "Synthetic", "given": [p]}]} for p in (pid, other)]
        # Include unrelated preexisting state so collateral-change checks are meaningful.
        base.append(observation(other, "unrelated", "2345-7", 95, "mg/dL", OLDER))
        t = {"id": f"syn-{i:03d}", "family": family, "patient": pid, "distractor": other,
             "clock": NOW, "provenance": VERSION, "split": "development", "seed": seed,
             "read_code": None, "write_type": None, "contract": {"required_creations": []}}
        if family in {"latest_lab", "conditional_medication"}:
            code, unit = ("2823-3", "mmol/L") if family == "latest_lab" else ("19123-9", "mg/dL")
            # Balance medication action/no-action branches by construction, before seeing models.
            needs_action = (i // 4) % 2 == 0
            value = round(rng.uniform(2.8, 4.8), 1) if family == "latest_lab" else (
                rng.choice([1.2, 1.3, 1.4]) if needs_action else rng.choice([2.0, 2.1, 2.2]))
            alternate = round(value + 1.1, 1) if family == "latest_lab" else (2.1 if needs_action else 1.3)
            base.extend([observation(pid, "lab-old", code, alternate, unit, OLDER),
                         observation(pid, "lab-latest", code, value, unit, LATEST),
                         observation(other, "lab-other", code, alternate, unit, LATEST)])
            t["read_code"] = code
            if family == "latest_lab":
                t["prompt"] = (f"For Patient/{pid}, return the most recent final laboratory value with LOINC {code} "
                    f"at or before {NOW}. Return final answer as {{\"value\": number, \"unit\": \"{unit}\"}}. "
                    "Do not change any record.")
                t["contract"]["answer"] = {"value": value, "unit": unit}
            else:
                t["write_type"] = "MedicationRequest"
                medication = {"resourceType": "MedicationRequest", "status": "active", "intent": "order",
                    "subject": {"reference": f"Patient/{pid}"},
                    "medicationCodeableConcept": concept("http://hl7.org/fhir/sid/ndc", "0338-1715-40"),
                    "authoredOn": NOW, "dosageInstruction": [{"route": concept("http://snomed.info/sct", "47625008"),
                    "doseAndRate": [{"doseQuantity": {"value": 2, "unit": "g"}}],
                    "timing": {"repeat": {"duration": 2, "durationUnit": "h"}}}]}
                t["action_template"] = medication
                t["prompt"] = (f"For Patient/{pid}, retrieve the most recent final magnesium (LOINC {code}) "
                    f"within the last 24 hours, as of {NOW}. Apply this research protocol exactly: if the value "
                    "is below 1.5 mg/dL, create exactly one MedicationRequest using this resource: " +
                    str_json(medication) + ". Otherwise create nothing. Return final answer as "
                    '{"action":"ordered"} or {"action":"none"} according to the protocol. '
                    "Do not change any other records. This is a synthetic protocol, not clinical guidance.")
                t["contract"]["answer"] = {"action": "ordered" if needs_action else "none"}
                if needs_action:
                    t["contract"]["required_creations"] = [medication]
        elif family == "record_vital":
            value = rng.randint(60, 95)
            r = observation(pid, "unused", "8867-4", value, "/min", NOW)
            r.pop("id")
            t["write_type"] = "Observation"
            t["action_template"] = r
            t["prompt"] = (f"Record a newly measured heart rate of {value} /min for Patient/{pid} at {NOW}. "
                "Create exactly one Observation with this resource: " + str_json(r) +
                '. Do not change anything else. Return final answer as {"action":"recorded"}.')
            t["contract"] = {"answer": {"action": "recorded"}, "required_creations": [r]}
        else:
            r = {"resourceType": "ServiceRequest", "status": "active", "intent": "order",
                 "subject": {"reference": f"Patient/{pid}"}, "authoredOn": NOW,
                 "code": concept("http://snomed.info/sct", "306181000000106"),
                 "note": [{"text": "Evaluate documented knee injury; synthetic research referral."}]}
            t["write_type"] = "ServiceRequest"
            t["action_template"] = r
            t["prompt"] = (f"For Patient/{pid}, create exactly one orthopedic referral with this resource: " +
                str_json(r) + '. Do not change any other records. Return final answer as {"action":"referred"}.')
            t["contract"] = {"answer": {"action": "referred"}, "required_creations": [r]}
        t["resources"] = copy.deepcopy(base)
        t["hazards"] = ["clean", "transient_503"]
        if t["read_code"]:
            t["hazards"] += ["stale", "patient_mismatch"]
        if t["contract"]["required_creations"]:
            t["hazards"] += ["commit_timeout", "no_commit_timeout"]
        tasks.append(t)
    return tasks


def str_json(value):
    from .storage import stable_json
    return stable_json(value)
