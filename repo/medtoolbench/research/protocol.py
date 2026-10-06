"""Public protocol shared across local systems; fixed before held-out runs."""
from medtoolbench.run_local import parse_action as parse_json_action
from .transport import DESCRIPTION
from .input_validation import validate_body

SYSTEM = '''You operate an isolated research FHIR R4 sandbox. No actual patient care is performed.
Follow the task exactly. Use exactly one JSON object per turn:
{"method":"GET","path":"/Observation?patient=P123&code=K&_sort=-date&_count=1"}
or {"method":"POST","path":"/ResourceType","body":{...}}
or {"method":"PUT","path":"/ResourceType/id","body":{...}}
or {"method":"DELETE","path":"/ResourceType/id"}
or {"final":{...the answer object requested in the task...}}.
Only the first JSON object is executed; wait for its real response. Do not invent results.
Tools return a user message with status and body. Resource types: Patient, Observation,
MedicationRequest, ServiceRequest. GET /metadata describes the current interface.
GET /Patient/id retrieves a patient. Patient search: birthdate, family, given, name;
name matches name parts, not necessarily a concatenated full name. Labs use local codes
MG, GLU, K, A1C exactly as supplied by the original dataset. No canonical-LOINC claim
is made for these local assay identifiers. Observation search: patient or subject,
code, status, date=geTIMESTAMP and date=leTIMESTAMP, _sort=-date, _count=N.
Multiple date filters can be combined. Status values final,amended,corrected are finalized.
Read values from entry[].resource.valueQuantity, dates from effectiveDateTime, and
identity from subject.reference. Follow link[relation=next] when more results are needed.
Relative paths only. For query dates, URL-encode '+' or use Z. Frozen time is
2023-11-13T10:15:00+00:00. Do not use the server's wall clock for task decisions.

Creation serialization profile (use the requested values; do not invent extra fields):
- All orders: status="active", intent="order", subject={"reference":"Patient/MRN"},
  authoredOn="2023-11-13T10:15:00+00:00". Do not supply id or meta on POST.
- ServiceRequest: code={"coding":[{"system":"http://loinc.org","code":"requested code"}]}.
  Referrals use system "http://snomed.info/sct" and note=[{"text":"exact requested note"}].
  Scheduled lab orders add occurrenceDateTime="requested ISO timestamp"; omit when unspecified.
- MedicationRequest: medicationCodeableConcept={"coding":[{"system":"http://hl7.org/fhir/sid/ndc","code":"requested NDC"}]}.
  dosageInstruction=[{"route":{"text":"Intravenous or Oral"},
  "timing":{"repeat":{"count":1}},"doseAndRate":[{"doseQuantity":{"value":NUMBER,"unit":"g or mEq"}}]}].
  For an infusion, add duration=NUMBER and durationUnit="h" inside timing.repeat.
- BP Observation: status="final", code={"text":"BP"}, subject={"reference":"Patient/MRN"},
  effectiveDateTime="2023-11-13T10:15:00+00:00", component=[
  {"code":{"coding":[{"system":"http://loinc.org","code":"8480-6"}]},"valueQuantity":{"value":SYSTOLIC,"unit":"mm[Hg]"}},
  {"code":{"coding":[{"system":"http://loinc.org","code":"8462-4"}]},"valueQuantity":{"value":DIASTOLIC,"unit":"mm[Hg]"}}].
FHIR permits more representations, but this benchmark uses the explicit profiles above.
''' + DESCRIPTION

WARNING = "\nTools may occasionally return unreliable or ambiguous results. Verify important observations and state-changing actions when appropriate."
HINTS = {"clean": "No injected tool hazard is present.", "interface_drift": "A documented search parameter has changed; discover the active interface.",
         "transient_503": "A transient execution failure is present.", "stale": "An observation response may be stale or incomplete.",
         "patient_mismatch": "A returned observation may belong to a different patient.", "provenance_conflict": "A non-authoritative preliminary cache response may disagree with the source of record.",
         "commit_timeout": "A write response may be lost; the request's commit status is unknown.",
         "no_commit_timeout": "A write response may be lost; the request's commit status is unknown."}


def messages(task, condition, hazard):
    system = SYSTEM
    if condition == "warning":
        system += WARNING
    if condition == "oracle_hint":
        system += "\nHazard-class hint (not a solution): " + HINTS[hazard]
    return [{"role": "system", "content": system}, {"role": "user", "content": task["prompt"]}]


def parse_action(raw):
    # Only designated final-channel text is executable. Truncated reasoning is
    # not scanned for conveniently valid actions. All raw tokens remain logged.
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    elif "<think>" in raw:
        raise ValueError("Incomplete reasoning block; no action dispatched")
    action = parse_json_action(raw.strip())
    if "final" not in action:
        validate_body(action.get("body"))
    return action
