"""Complete patient-history transfer, using import receipts, never gold evidence IDs."""
import gzip
import json
from collections import defaultdict

from .fhir import normalized_state

KINDS = ("Patient", "Encounter", "Observation")


def patient_records(db, patient):
    rows = [json.loads(gzip.decompress(blob)) for (blob,) in db.execute(
        "SELECT body_gzip FROM resources WHERE (kind='Patient' AND id=?) OR (patient=? AND kind IN ('Encounter','Observation')) ORDER BY rowid",
        (patient, patient))]
    return sorted(rows, key=lambda r: (0 if r["resourceType"] == "Patient" else
                  2 if r["resourceType"] == "Encounter" and r.get("partOf") else
                  1 if r["resourceType"] == "Encounter" else 3, r["id"]))


def import_records(store, resources, progress=None):
    receipts = defaultdict(list)
    for offset in range(0, len(resources), 200):
        batch = resources[offset:offset + 200]
        store.transaction([{"request": {"method": "PUT", "url": r["resourceType"] + "/" + r["id"]},
                            "resource": r} for r in batch])
        for resource in batch:
            receipts[resource["resourceType"]].append(resource["id"])
        if progress:
            progress(min(offset + 200, len(resources)))
    return dict(receipts)


def fetch_kind(store, kind, ids):
    """Bounded _id queries cover the complete import receipt, not answer-relevant IDs.

    This is a declared retrieval SDK adaptation, not a claim that the upstream
    server's unconstrained search returns arbitrarily long patient histories.
    """
    if kind not in KINDS or len(ids) != len(set(ids)):
        raise ValueError("Invalid kind or duplicate import receipt")
    resources = []
    for offset in range(0, len(ids), 200):
        wanted = ids[offset:offset + 200]
        batch = store.all_pages(f"/{kind}?_id=" + ",".join(wanted) + "&_count=200&_total=accurate")
        if len(batch) != len(wanted) or {r["id"] for r in batch} != set(wanted):
            raise RuntimeError("Truncated or unexpected FHIR retrieval; do not score incomplete patient history")
        if any(r["resourceType"] != kind for r in batch):
            raise RuntimeError("FHIR response resource type mismatch")
        resources.extend(batch)
    return resources


def verified_roundtrip(store, resources, receipts):
    returned = [r for kind in KINDS for r in fetch_kind(store, kind, receipts.get(kind, []))]
    if normalized_state(resources) != normalized_state(returned):
        raise RuntimeError("Complete imported FHIR history differs from HTTP-returned resources")
    return returned
