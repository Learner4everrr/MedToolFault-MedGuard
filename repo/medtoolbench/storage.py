"""Small persistent FHIR-shaped store. NOT a complete FHIR implementation."""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from urllib.parse import parse_qs, urlsplit


def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def outcome(status, message):
    return {"status": status, "body": {"resourceType": "OperationOutcome", "issue": [
        {"severity": "error", "code": "processing", "diagnostics": message}]}}


def codes(resource):
    c = resource.get("code", resource.get("medicationCodeableConcept", {}))
    return [x.get("code") for x in c.get("coding", [])]


def subject(resource):
    return resource.get("subject", {}).get("reference")


def bundle(resources):
    return {"resourceType": "Bundle", "type": "searchset", "total": len(resources),
            "entry": [{"resource": copy.deepcopy(r)} for r in resources]}


class Store:
    """All agent writes commit to SQLite before their response is returned."""

    def __init__(self, path, resources):
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS resources (kind TEXT, id TEXT, body TEXT, PRIMARY KEY(kind,id))")
        self.initial = copy.deepcopy(resources)
        self.counter = 0
        self.reset()

    def reset(self):
        with self.db:
            self.db.execute("DELETE FROM resources")
            for r in self.initial:
                self.db.execute("INSERT INTO resources VALUES(?,?,?)",
                                (r["resourceType"], r["id"], stable_json(r)))
        self.counter = 0
        assert digest(self.snapshot()) == digest(sorted(self.initial, key=lambda r: (r["resourceType"], r["id"])))

    def snapshot(self):
        return [json.loads(x[0]) for x in self.db.execute("SELECT body FROM resources ORDER BY kind,id")]

    def close(self):
        self.db.close()

    def call(self, method, path, body=None):
        parsed = urlsplit(path)
        if parsed.scheme or parsed.netloc or not path.startswith("/") or ".." in path:
            return outcome(400, "Only relative sandbox resource paths are allowed")
        segments = parsed.path.strip("/").split("/")
        kind = segments[0]
        q = parse_qs(parsed.query)
        if method == "GET" and kind == "metadata":
            return {"status": 200, "body": {"resourceType": "CapabilityStatement",
                "status": "active", "date": "2023-11-13", "kind": "instance", "fhirVersion": "4.0.1",
                "format": ["json"], "description": "Research subset: read, search, create, update, delete. "
                "Search: patient/subject, code, status, date=ge/le, _sort=-date, _count. "
                "Omit _count to retrieve all matches; reads include prior creations."}}
        if kind not in {"Patient", "Observation", "MedicationRequest", "ServiceRequest"}:
            return outcome(404, "Unsupported resource type")
        rows = [r for r in self.snapshot() if r["resourceType"] == kind]
        if method == "GET":
            if len(segments) == 2:
                found = [r for r in rows if r["id"] == segments[1]]
                return {"status": 200, "body": found[0]} if found else outcome(404, "Not found")
            if len(segments) != 1:
                return outcome(400, "Unsupported path")
            allowed = {"patient", "subject", "code", "status", "date", "_sort", "_count", "_id"}
            if set(q) - allowed:
                return outcome(400, "Unsupported search parameter; GET /metadata for capabilities")
            for key, values in q.items():
                if key in {"subject", "patient"}:
                    ref = values[0] if "/" in values[0] else "Patient/" + values[0]
                    rows = [r for r in rows if subject(r) == ref]
                elif key == "code":
                    rows = [r for r in rows if values[0].split("|")[-1] in codes(r)]
                elif key in {"status", "_id"}:
                    rows = [r for r in rows if r.get("id" if key == "_id" else key) == values[0]]
                elif key == "date":
                    for value in values:
                        if not value.startswith(("ge", "le")):
                            return outcome(400, "date supports ge/le ISO timestamps")
                        rows = [r for r in rows if (r.get("effectiveDateTime", "") >= value[2:]
                            if value[:2] == "ge" else r.get("effectiveDateTime", "") <= value[2:])]
            if "_sort" in q:
                if q["_sort"][0] not in {"date", "-date"}:
                    return outcome(400, "Only date sorting is supported")
                rows.sort(key=lambda r: (r.get("effectiveDateTime", ""), r["id"]),
                          reverse=q["_sort"][0].startswith("-"))
            total = len(rows)
            if "_count" in q:
                try:
                    count = int(q["_count"][0])
                except ValueError:
                    return outcome(400, "_count must be an integer")
                if count < 1:
                    return outcome(400, "_count must be positive")
                rows = rows[:count]
            b = bundle(rows)
            b["total"] = total
            return {"status": 200, "body": b}
        if method == "DELETE" and len(segments) == 2:
            with self.db:
                self.db.execute("DELETE FROM resources WHERE kind=? AND id=?", (kind, segments[1]))
            return {"status": 204, "body": {}}
        if method not in {"POST", "PUT"}:
            return outcome(405, "Unsupported method")
        if not isinstance(body, dict) or body.get("resourceType") != kind:
            return outcome(400, "body.resourceType must match path")
        if kind != "Patient" and not subject(body):
            return outcome(400, "subject.reference is required")
        if method == "POST" and len(segments) != 1 or method == "PUT" and len(segments) != 2:
            return outcome(400, "POST /Type or PUT /Type/id required")
        r = copy.deepcopy(body)
        if method == "POST":
            self.counter += 1
            r["id"] = f"created-{self.counter:06}"
            while any(x["id"] == r["id"] for x in rows):
                self.counter += 1
                r["id"] = f"created-{self.counter:06}"
        else:
            r["id"] = segments[1]
        r["meta"] = {"versionId": "1", "lastUpdated": "2023-11-13T10:15:00Z"}
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO resources VALUES(?,?,?)", (kind, r["id"], stable_json(r)))
        return {"status": 201 if method == "POST" else 200, "body": r}
