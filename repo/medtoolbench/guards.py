"""MedGuard: public-interface checks, no gold answers or hidden hazard access."""
from urllib.parse import parse_qs, urlencode, urlsplit

from .storage import bundle, codes, subject


def equivalent(left, right):
    # Separate from grader: whole payload comparison after transport-generated fields.
    def payload(r):
        return {k: v for k, v in r.items() if k not in {"id", "meta", "text"}}
    return payload(left) == payload(right)


class Client:
    def __init__(self, proxy, condition="vanilla"):
        self.proxy, self.condition = proxy, condition

    def raw(self, method, path, body=None, actor="agent"):
        return self.proxy.call(method, path, body, actor)

    def call(self, method, path, body=None):
        method = method.upper()
        result = self.raw(method, path, body)
        if self.condition == "retry_only":
            for _ in range(2):
                if result["status"] < 500:
                    break
                result = self.raw(method, path, body, "retry_only")
            return result
        if self.condition != "medguard":
            return result
        if method == "GET":
            for _ in range(2):
                if result["status"] < 500:
                    break
                result = self.raw(method, path, body, "medguard")
            parsed = urlsplit(path)
            q = parse_qs(parsed.query)
            if parsed.path == "/Observation" and ("patient" in q or "subject" in q) and "code" in q:
                # Deliberately conservative cost-bearing policy: verify all coded observation
                # searches by fetching the full matching history and reapplying sort/limit.
                verify_q = {k: v for k, v in q.items() if k != "_count"}
                fresh = self.raw("GET", "/Observation?" + urlencode(verify_q, doseq=True), actor="medguard")
                if fresh["status"] == 200:
                    ref = q.get("patient", q.get("subject"))[0]
                    if "/" not in ref:
                        ref = "Patient/" + ref
                    rows = [e["resource"] for e in fresh["body"].get("entry", [])]
                    rows = [r for r in rows if subject(r) == ref and q["code"][0] in codes(r)]
                    if "_sort" in q:
                        rows.sort(key=lambda r: (r.get("effectiveDateTime", ""), r["id"]),
                                  reverse=q["_sort"][0].startswith("-"))
                    if "_count" in q:
                        try:
                            rows = rows[:int(q["_count"][0])]
                        except ValueError:
                            return result
                    result = {"status": 200, "body": bundle(rows)}
            return result
        if method == "POST" and result["status"] >= 500 and isinstance(body, dict):
            ref = subject(body)
            if not ref:
                return result
            found = self.raw("GET", path.split("?")[0] + "?subject=" + ref, actor="medguard")
            if found["status"] != 200:
                return result  # Inconclusive verification: never blindly retry.
            entries = [e["resource"] for e in found["body"].get("entry", [])]
            matching = [r for r in entries if equivalent(r, body)]
            if matching:
                return {"status": 200, "body": matching[0]}
            return self.raw(method, path, body, "medguard")
        return result
