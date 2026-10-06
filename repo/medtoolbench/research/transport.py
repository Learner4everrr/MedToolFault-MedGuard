"""Observable, recoverable hazards and public-interface-only guard controls."""
import copy
import time
from urllib.parse import parse_qs, urlencode, urlsplit

from medtoolbench.storage import bundle, codes, outcome, subject
from .contracts import clinical, instant
from .input_validation import validate_path

DESCRIPTION = ("Owned research FHIR R4 sandbox. Search is paginated: follow link[relation=next]. "
    "Final/amended/corrected authoritative results take precedence over preliminary cache responses. "
    "GET /authoritative/Observation uses the source-of-record read endpoint with the same search parameters. "
    "All successful writes are immediately searchable in this controlled single-writer experiment. "
    "POST creates a new resource; verify a response-ambiguous write against a pre-write snapshot before retrying.")


class Proxy:
    def __init__(self, store, task, hazard="clean"):
        if hazard not in task["hazards"]:
            raise ValueError("Hazard was not declared eligible at construction")
        self.store, self.task, self.hazard = store, task, hazard
        self.triggered, self.trace, self.mutation_states = 0, [], []

    def metadata(self):
        # Wrapper capability metadata: intentionally a declared subset of HAPI,
        # not a fabricated complete server CapabilityStatement.
        params = ["subject", "code", "status", "date", "_sort", "_count"]
        if self.hazard != "interface_drift":
            params.insert(0, "patient")
        return {"status": 200, "body": {"resourceType": "CapabilityStatement", "status": "active",
            "date": "2023-11-13", "kind": "capability", "fhirVersion": "4.0.1", "format": ["json"],
            "description": DESCRIPTION, "rest": [{"mode": "server", "resource": [
                {"type": "Observation", "searchParam": [{"name": k, "type": "reference" if k in {"subject", "patient"} else "string"} for k in params]}]}]}}

    def call(self, method, path, body=None, actor="agent"):
        start = time.perf_counter()
        method, parsed = method.upper(), urlsplit(path)
        q = parse_qs(parsed.query)
        authoritative = parsed.path == "/authoritative/Observation" and method == "GET"
        routed = path.replace("/authoritative/Observation", "/Observation", 1) if authoritative else path
        kind = urlsplit(routed).path.strip("/").split("/")[0]
        ref = q.get("patient", q.get("subject", [None]))[0]
        target_read = method == "GET" and kind == "Observation" and ref in {self.task["patient"], f"Patient/{self.task['patient']}"}
        target_read = target_read and self.task.get("read_code") in [v.split("|")[-1] for v in q.get("code", [])]
        target_write = method == "POST" and isinstance(body, dict) and subject(body) == f"Patient/{self.task['patient']}"
        target_write = target_write and kind in {r["resourceType"] for r in self.task["contract"]["required_creations"]}
        fire, result, committed = False, None, None
        if method == "GET" and parsed.path == "/metadata":
            result = self.metadata()
        elif self.hazard == "interface_drift" and method == "GET" and kind == "Observation" and "patient" in q:
            fire = True
            result = outcome(400, "Unsupported search parameter 'patient'; consult GET /metadata for the active interface.")
        elif not self.triggered:
            if self.hazard == "transient_503" and kind in self.store.types and method in {"GET", "POST"}:
                fire, result = True, outcome(503, "Service temporarily unavailable")
            elif self.hazard in {"commit_timeout", "no_commit_timeout"} and target_write:
                fire = True
                if self.hazard == "commit_timeout":
                    committed = self.store.call(method, routed, body)
                    if committed["status"] >= 400:
                        fire, result = False, committed
                if result is None:
                    result = outcome(504, "Gateway timeout: request outcome unknown")
            elif target_read and not authoritative:
                if self.hazard == "stale":
                    fire = True
                    result = {"status": 200, "body": bundle([self.task["stale_resource"]])}
                elif self.hazard == "patient_mismatch":
                    fire = True
                    result = {"status": 200, "body": bundle(self.task["distractor_resources"])}
                elif self.hazard == "provenance_conflict":
                    original = self.store.call(method, routed, body)
                    rows = [e["resource"] for e in original["body"].get("entry", []) if "valueQuantity" in e.get("resource", {})]
                    if original["status"] == 200 and rows:
                        r = copy.deepcopy(rows[0])
                        r["status"] = "preliminary"
                        r.setdefault("meta", {})["source"] = "urn:medtoolbench:non-authoritative-cache"
                        r["valueQuantity"]["value"] = round(float(r["valueQuantity"]["value"]) * 1.4 + .7, 6)
                        result = {"status": 200, "body": bundle([r])}
                        result["body"]["link"] = [{"relation": "related", "url": "/authoritative/Observation?" + parsed.query}]
                        fire = True
        if result is None:
            result = self.store.call(method, routed, body)
        result = copy.deepcopy(result)
        for link in result["body"].get("link", []):
            if link["url"].startswith(self.store.base + "/"):
                link["url"] = link["url"][len(self.store.base):]
            elif link["url"].startswith(self.store.base + "?"):
                link["url"] = "/" + link["url"][len(self.store.base):]
        if fire:
            self.triggered += 1
        if method in {"POST", "PUT", "DELETE"} and (result["status"] < 400 or committed and committed["status"] < 400):
            self.mutation_states.append({"trace_index": len(self.trace), "state": self.store.snapshot()})
        self.trace.append({"method": method, "path": path, "body": copy.deepcopy(body), "actor": actor,
            "response": result, "injected": fire, "committed_response": committed,
            "wall_seconds": time.perf_counter() - start})
        return copy.deepcopy(result)


class Client:
    """Only call(method,path,body), not task, hazard, source data or gold contracts."""
    def __init__(self, call, condition="vanilla"):
        if condition not in {"vanilla", "warning", "retry_only", "medguard", "oracle_hint"}:
            raise ValueError(condition)
        self.raw, self.condition = call, condition

    def read(self, path):
        result = self.raw("GET", path, actor="medguard")
        for _ in range(2):
            if result["status"] < 500:
                break
            result = self.raw("GET", path, actor="medguard")
        if result["status"] == 400 and "patient" in parse_qs(urlsplit(path).query):
            meta = self.raw("GET", "/metadata", actor="medguard")
            resources = [r for rest in meta["body"].get("rest", []) for r in rest.get("resource", []) if r.get("type") == "Observation"]
            if any(p.get("name") == "subject" for r in resources for p in r.get("searchParam", [])):
                parsed = urlsplit(path)
                q = parse_qs(parsed.query)
                q["subject"] = q.pop("patient")
                result = self.raw("GET", parsed.path + "?" + urlencode(q, doseq=True), actor="medguard")
        return result

    def pages(self, path):
        result = self.read(path)
        if result["status"] != 200 or result["body"].get("resourceType") != "Bundle":
            return result
        rows, visited = [], set()
        while True:
            rows.extend(e["resource"] for e in result["body"].get("entry", []) if e.get("search", {}).get("mode", "match") == "match")
            path = next((x["url"] for x in result["body"].get("link", []) if x["relation"] == "next"), None)
            if not path:
                return {"status": 200, "body": bundle(rows)}
            if path in visited or not path.startswith("/") or urlsplit(path).netloc:
                return outcome(502, "Untrusted or cyclic pagination link")
            visited.add(path)
            result = self.read(path)
            if result["status"] != 200:
                return result

    def matching(self, path, body):
        response = self.pages(path.split("?")[0] + "?" + urlencode({"subject": subject(body), "_count": 2000}))
        if response["status"] != 200:
            return None
        return {r["id"]: r for e in response["body"].get("entry", [])
                if (r := e["resource"]) and clinical(r) == clinical(body)}

    def call(self, method, path, body=None):
        method = method.upper()
        try:
            validate_path(path)
        except ValueError:
            # Do not let a guard's preflight repair malformed syntax implicitly.
            # The raw transport records and returns the same 400 to every policy.
            return self.raw(method, path, body)
        prior = None
        if self.condition == "medguard" and method == "POST" and isinstance(body, dict) and subject(body):
            prior = self.matching(path, body)
        result = self.raw(method, path, body)
        if self.condition == "retry_only":
            for _ in range(2):
                if result["status"] < 500:
                    break
                result = self.raw(method, path, body, actor="retry_only")
            return result
        if self.condition != "medguard":
            return result
        if method == "GET":
            parsed, q = urlsplit(path), parse_qs(urlsplit(path).query)
            if parsed.path in {"/Observation", "/authoritative/Observation"} and "code" in q and ("patient" in q or "subject" in q):
                requested_count = int(q["_count"][0]) if "_count" in q else None
                q.pop("_count", None)
                if "patient" in q:
                    q["subject"] = q.pop("patient")
                fresh = self.pages("/authoritative/Observation?" + urlencode(q, doseq=True))
                if fresh["status"] == 200:
                    patient = q["subject"][0].removeprefix("Patient/")
                    wanted_codes = {c.split("|")[-1] for c in q["code"]}
                    rows = [e["resource"] for e in fresh["body"].get("entry", [])]
                    rows = [r for r in rows if subject(r) == f"Patient/{patient}" and wanted_codes.intersection(codes(r))
                            and r.get("status") in {"final", "amended", "corrected"}]
                    if "_sort" in q:
                        rows.sort(key=lambda r: (instant(r["effectiveDateTime"]), r["id"]), reverse=q["_sort"][0].startswith("-"))
                    if requested_count is not None:
                        rows = rows[:requested_count]
                    return {"status": 200, "body": bundle(rows)}
                # Verification failed: do not pass through the unchecked answer
                # as though identity/freshness had been established.
                return outcome(502, "Authoritative observation verification incomplete; no verified result returned")
            if result["status"] >= 500 or result["status"] == 400:
                return self.read(path)
        if method == "POST" and result["status"] >= 500 and prior is not None:
            after = self.matching(path, body)
            if after is None:
                return result
            created = after.keys() - prior.keys()
            if len(created) == 1:
                return {"status": 200, "body": after[next(iter(created))]}
            if not created:
                return self.raw(method, path, body, actor="medguard")
            return outcome(409, "Multiple equivalent writes appeared; cannot attribute the ambiguous request")
        return result
