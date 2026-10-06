"""Public-interface controls and explicitly parameterized fault stress tests."""
import copy
from urllib.parse import parse_qs, urlencode, urlsplit

from medtoolbench.storage import outcome, subject
from medtoolbench.research.transport import Client, Proxy
from medtoolbench.research.input_validation import validate_path

CONDITIONS = ("vanilla", "informed", "evidence_only", "medguard", "no_read", "no_write", "no_retry")
INFORMED = """
Before acting, verify patient identity, assay code, finalized status and task time window.
Use the documented authoritative observation endpoint to check important reads.
Before a POST, record existing clinically equivalent resources. After a timeout,
compare a new read with that snapshot: one new equivalent resource can establish
commit only under the stated single-writer, immediately-visible assumptions.
Do not blindly repeat a write. Consult /metadata after an interface error.
"""


class CallBudget:
    """Counts every raw public operation, including automatic helper calls."""
    def __init__(self, call, maximum=48):
        self.raw, self.maximum, self.used, self.denied = call, maximum, 0, 0

    def __call__(self, method, path, body=None, actor="agent"):
        if self.used >= self.maximum:
            self.denied += 1
            return outcome(429, "This episode's shared raw-tool-call budget is exhausted")
        self.used += 1
        return self.raw(method, path, body, actor=actor)


class FactorClient(Client):
    """Component deletion, not a new optimized guard. Has no task or gold access."""
    def __init__(self, call, condition):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        super().__init__(call, "medguard")
        self.policy = condition

    def read(self, path):
        if self.policy == "no_retry":
            # Remove retries and metadata-based interface repair together.
            return self.raw("GET", path, actor="medguard")
        return super().read(path)

    def call(self, method, path, body=None):
        method = method.upper()
        if self.policy in {"vanilla", "informed"}:
            return self.raw(method, path, body)
        try:
            validate_path(path)
        except ValueError:
            return self.raw(method, path, body)
        if self.policy == "no_read" and method == "GET":
            result = self.raw(method, path, body)
            return self.read(path) if result["status"] >= 400 else result
        if self.policy == "no_write" and method == "POST":
            result = self.raw(method, path, body)
            # Keep generic retry; remove commit-attribution/pre-write snapshot.
            for _ in range(2):
                if result["status"] < 500:
                    break
                result = self.raw(method, path, body, actor="retry_only")
            return result
        if self.policy == "evidence_only":
            prior = None
            if method == "POST" and isinstance(body, dict) and subject(body):
                prior = self.matching(path, body)
            original = self.raw(method, path, body)
            parsed, q = urlsplit(path), parse_qs(urlsplit(path).query)
            evidence = {}
            if method == "GET" and parsed.path in {"/Observation", "/authoritative/Observation"} and "code" in q and ("patient" in q or "subject" in q):
                q.pop("_count", None)
                if "patient" in q:
                    q["subject"] = q.pop("patient")
                evidence["source_of_record_response"] = self.pages("/authoritative/Observation?" + urlencode(q, doseq=True))
            if method == "POST" and prior is not None:
                evidence["equivalent_records_before"] = list(prior.values())
                if original["status"] >= 500:
                    after = self.matching(path, body)
                    evidence["equivalent_records_after"] = None if after is None else list(after.values())
            if evidence:
                # Raw evidence is supplied without filtering, choosing a winner,
                # asserting commit, or automatically repeating the write.
                return {**original, "verification_evidence": evidence}
            return original
        if self.policy == "no_retry" and method == "POST":
            prior = self.matching(path, body) if isinstance(body, dict) and subject(body) else None
            result = self.raw(method, path, body)
            if result["status"] >= 500 and prior is not None:
                after = self.matching(path, body)
                if after is not None and len(after.keys() - prior.keys()) == 1:
                    return {"status": 200, "body": after[next(iter(after.keys() - prior.keys()))]}
            return result
        return super().call(method, path, body)


class StressProxy(Proxy):
    """Replay existing fault semantics at declared operations, plus boundary arms.

    Component triggers are tracked separately. `agent_target` never lets guard
    preflight consume a transient. Delay/concurrency are boundary violations,
    not declared recoverable cases, and remain in the output denominator.
    """
    def __init__(self, store, task, hazard, spec=None):
        spec = spec or {}
        self.components = hazard.split("+")
        if any(h not in task["hazards"] for h in self.components):
            raise ValueError("Ineligible component fault")
        super().__init__(store, task, self.components[0])
        self.spec = spec
        self.counts = {h: 0 for h in self.components}
        self.hidden_ids, self.external_ids = set(), set()
        self.delay_left = int(spec.get("visibility_delay", 0))
        self.concurrent_done = False

    def target(self, method, path, body):
        q = parse_qs(urlsplit(path).query)
        kind = urlsplit(path).path.strip("/").split("/")[-1]
        if method == "GET":
            ref = q.get("patient", q.get("subject", [None]))[0]
            if self.task.get("read_code"):
                return kind == "Observation" and ref in {self.task["patient"], "Patient/" + self.task["patient"]} and self.task["read_code"] in [c.split("|")[-1] for c in q.get("code", [])]
            return kind == "Patient" or path == "/Patient/" + self.task["patient"]
        return method == "POST" and isinstance(body, dict) and subject(body) == "Patient/" + self.task["patient"] and kind in {r["resourceType"] for r in self.task["contract"]["required_creations"]}

    def call(self, method, path, body=None, actor="agent"):
        method = method.upper()
        total = self.triggered
        chosen = "clean"
        repeats = int(self.spec.get("repeat", 1))
        for h in self.components:
            if h == "clean":
                continue
            if h != "interface_drift" and self.counts[h] >= repeats:
                continue
            if h == "transient_503" and self.spec.get("placement") == "agent_target" and (actor != "agent" or not self.target(method, path, body)):
                continue
            # Select a relevant component without swallowing later components.
            if h in {"stale", "patient_mismatch", "provenance_conflict"} and (method != "GET" or not self.target(method, path, body) or path.startswith("/authoritative/")):
                continue
            if h in {"commit_timeout", "no_commit_timeout"} and (method != "POST" or not self.target(method, path, body)):
                continue
            if h == "interface_drift" and not (method == "GET" and "patient" in parse_qs(urlsplit(path).query) and "Observation" in path):
                continue
            chosen = h
            break
        other = None
        if chosen in {"commit_timeout", "no_commit_timeout"} and self.spec.get("concurrent") and not self.concurrent_done:
            other = self.store.call(method, path, body)
            if other["status"] >= 400:
                raise RuntimeError("Controlled concurrent write failed")
            self.concurrent_done = True
            self.external_ids.add(other["body"]["id"])
        self.hazard, self.triggered = chosen, 0
        result = super().call(method, path, body, actor)
        fired = bool(self.trace[-1]["injected"])
        if fired:
            self.counts[chosen] += 1
        self.triggered = total + int(fired)
        record = self.trace[-1]
        record["fault_component"] = chosen if fired else None
        if other:
            record["external_committed_response"] = other
            # Final-state score includes all new records; attribution is separate.
            if not self.mutation_states or self.mutation_states[-1]["trace_index"] != len(self.trace) - 1:
                self.mutation_states.append({"trace_index": len(self.trace) - 1, "state": self.store.snapshot()})
        committed = record.get("committed_response")
        if committed and committed["status"] < 400:
            self.hidden_ids.add(committed["body"]["id"])
        if method == "GET" and self.delay_left and self.hidden_ids and result["body"].get("resourceType") == "Bundle":
            entries = result["body"].get("entry", [])
            if any(e.get("resource", {}).get("id") in self.hidden_ids for e in entries):
                result = copy.deepcopy(result)
                result["body"]["entry"] = [e for e in entries if e.get("resource", {}).get("id") not in self.hidden_ids]
                result["body"]["total"] = len(result["body"]["entry"])
                self.delay_left -= 1
                record["response"] = copy.deepcopy(result)
                record["visibility_masked"] = True
        return result
