"""Deterministic, one-shot recoverable hazards. Hidden metadata stays in traces."""
import copy
from urllib.parse import parse_qs, urlsplit

from .storage import bundle, codes, outcome, subject


class Proxy:
    def __init__(self, store, task, hazard="clean"):
        self.store, self.task, self.hazard = store, task, hazard
        self.triggered = 0
        self.trace = []

    def call(self, method, path, body=None, actor="agent"):
        method = method.upper()
        parsed = urlsplit(path)
        kind = parsed.path.strip("/").split("/")[0]
        query = parse_qs(parsed.query)
        p = self.task["patient"]
        read_target = (method == "GET" and kind == "Observation" and
                       query.get("patient", query.get("subject", [None]))[0] in {p, f"Patient/{p}"} and
                       query.get("code", [None])[0] == self.task.get("read_code"))
        write_target = (method == "POST" and kind == self.task.get("write_type")
                        and isinstance(body, dict) and subject(body) == f"Patient/{p}")
        fire = self.triggered == 0 and (
            self.hazard == "transient_503" and (read_target or write_target) or
            self.hazard in {"stale", "patient_mismatch"} and read_target or
            self.hazard in {"commit_timeout", "no_commit_timeout"} and write_target)
        result = None
        if fire and self.hazard == "transient_503":
            result = outcome(503, "Service temporarily unavailable")
        elif fire and self.hazard in {"commit_timeout", "no_commit_timeout"}:
            if self.hazard == "commit_timeout":
                committed = self.store.call(method, path, body)
                if committed["status"] >= 400:
                    # Invalid writes are not falsely reported as committed.
                    fire = False
                    result = committed
            if result is None:
                result = outcome(504, "Gateway timeout: request outcome unknown")
        elif fire and self.hazard in {"stale", "patient_mismatch"}:
            pool = [r for r in self.store.snapshot() if r["resourceType"] == "Observation"
                    and self.task["read_code"] in codes(r)]
            if self.hazard == "stale":
                pool = [r for r in pool if subject(r) == f"Patient/{p}"]
                pool.sort(key=lambda r: r["effectiveDateTime"], reverse=True)
                injected = pool[1:2]
            else:
                injected = [r for r in pool if subject(r) == f"Patient/{self.task['distractor']}"][:1]
            if not injected:
                raise ValueError("Ineligible silent-hazard episode")
            result = {"status": 200, "body": bundle(injected)}
        if result is None:
            result = self.store.call(method, path, body)
        if fire:
            self.triggered += 1
        self.trace.append({"method": method, "path": path, "body": copy.deepcopy(body),
                           "actor": actor, "response": copy.deepcopy(result), "injected": bool(fire)})
        return result
