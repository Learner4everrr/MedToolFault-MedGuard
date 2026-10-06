"""Scripted reference uses public task fields, ordinary tools, no contract answers."""
from .guards import Client


def solve(task, proxy, condition="medguard"):
    client = Client(proxy, condition)
    family = task["family"]
    if task["read_code"]:
        path = (f"/Observation?patient={task['patient']}&code={task['read_code']}"
                f"&status=final&date=le{task['clock']}&_sort=-date&_count=1")
        response = client.call("GET", path)
        if response["status"] != 200:
            return {"error": "tool failure"}
        rows = [e["resource"] for e in response["body"].get("entry", [])]
        if not rows:
            return {"error": "missing observation"}
        value = rows[0]["valueQuantity"]
        if family == "latest_lab":
            return {"value": value["value"], "unit": value["unit"]}
        if value["value"] >= 1.5:
            return {"action": "none"}
    response = client.call("POST", "/" + task["write_type"], task["action_template"])
    if response["status"] >= 400:
        return {"error": "write failure"}
    return {"action": {"conditional_medication": "ordered", "record_vital": "recorded", "referral": "referred"}[family]}
