"""Real-HAPI mechanism/boundary experiments, explicitly synthetic and not patients.

Concurrency and delayed search visibility violate the primary single-writer,
immediate-visibility assumptions. Counterexamples are retained as outcomes, not
silently excluded or counted as failed primary benchmark construction.
"""
import argparse
import copy
import json
from pathlib import Path

from medtoolbench.storage import outcome
from .build import save_gzip
from .contracts import clinical, service
from .fhir import HapiProcess
from .job_state import write_json
from .transport import Client

CASES = ["commit_timeout", "no_commit_timeout", "preexisting_equivalent", "preexisting_different",
         "preexisting_cancelled", "visibility_delay_1", "verification_503_1", "verification_503_3",
         "concurrent_no_commit", "concurrent_commit"]


def experiment(store, case, condition):
    patient = "boundary-patient"
    payload = service(patient, "4548-4")
    fixture = [{"resourceType": "Patient", "id": patient}]
    if case.startswith("preexisting"):
        old = dict(copy.deepcopy(payload), id="previous-order")
        if case == "preexisting_different":
            old["code"]["coding"][0]["code"] = "2823-3"
        if case == "preexisting_cancelled":
            old["status"] = "revoked"
        fixture.append(old)
    before = store.reload(fixture)
    state = {"first_post": True, "masked": False, "agent_commits": 0, "other_commits": 0,
             "hidden_reads": 1 if case == "visibility_delay_1" else 0,
             "unavailable_reads": int(case.rsplit("_", 1)[1]) if case.startswith("verification_503") else 0,
             "new_ids": set()}
    trace = []
    def raw(method, path, body=None, actor="agent"):
        if method == "GET" and state["masked"] and state["unavailable_reads"]:
            state["unavailable_reads"] -= 1
            response = outcome(503, "Verification search temporarily unavailable")
        elif method == "POST":
            if state["first_post"]:
                state["first_post"], state["masked"] = False, True
                if case.startswith("concurrent"):
                    other = store.call(method, path, body)
                    if other["status"] >= 400:
                        raise RuntimeError(other)
                    state["other_commits"] += 1
                    state["new_ids"].add(other["body"]["id"])
                if case not in {"no_commit_timeout", "concurrent_no_commit"}:
                    committed = store.call(method, path, body)
                    if committed["status"] >= 400:
                        raise RuntimeError(committed)
                    state["agent_commits"] += 1
                    state["new_ids"].add(committed["body"]["id"])
                response = outcome(504, "Gateway timeout: request outcome unknown")
            else:
                response = store.call(method, path, body)
                if response["status"] < 400:
                    state["agent_commits"] += 1
                    state["new_ids"].add(response["body"]["id"])
        else:
            response = store.call(method, path, body)
            if method == "GET" and state["masked"] and state["hidden_reads"] and response["body"].get("resourceType") == "Bundle":
                state["hidden_reads"] -= 1
                response = copy.deepcopy(response)
                response["body"]["entry"] = [e for e in response["body"].get("entry", []) if e["resource"]["id"] not in state["new_ids"]]
                response["body"]["total"] = len(response["body"]["entry"])
        trace.append({"method": method, "path": path, "actor": actor, "body": body, "response": response})
        return response
    response = Client(raw, condition).call("POST", "/ServiceRequest", payload)
    after = store.snapshot()
    old_ids = {(r["resourceType"], r["id"]) for r in before}
    created = [r for r in after if (r["resourceType"], r["id"]) not in old_ids and r["resourceType"] == "ServiceRequest"]
    equal = [r for r in created if clinical(r) == clinical(payload)]
    result = {"case": case, "condition": condition, "response_status": response["status"],
              "new_equivalent_resources": len(equal), "agent_actual_commits": state["agent_commits"],
              "other_actual_commits": state["other_commits"], "duplicate_agent_commits": state["agent_commits"] > 1,
              "success_response_without_agent_commit": response["status"] < 400 and state["agent_commits"] == 0,
              "tool_calls": len(trace), "before": before, "after": after, "trace": trace}
    # Expected limitations are explicit positive checks of a counterexample.
    if condition == "medguard":
        if case == "visibility_delay_1":
            result["expected_behavior_observed"] = result["duplicate_agent_commits"]
        elif case == "concurrent_no_commit":
            result["expected_behavior_observed"] = result["success_response_without_agent_commit"]
        elif case == "concurrent_commit":
            result["expected_behavior_observed"] = response["status"] == 409 and state["agent_commits"] == 1
        elif case == "verification_503_3":
            result["expected_behavior_observed"] = response["status"] == 504 and state["agent_commits"] == 1
        else:
            result["expected_behavior_observed"] = response["status"] < 400 and state["agent_commits"] == 1
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Use a new boundary-experiment output directory")
    a.out.mkdir(parents=True)
    rows = []
    with HapiProcess(a.image, a.out / "hapi", heap="2g") as store:
        for case in CASES:
            for condition in ("vanilla", "retry_only", "medguard"):
                row = experiment(store, case, condition)
                save_gzip(a.out / (case + "--" + condition + ".json.gz"), row)
                rows.append({k: v for k, v in row.items() if k not in {"before", "after", "trace"}})
                print(json.dumps(rows[-1]), flush=True)
    report = {"episodes": len(rows), "results": rows, "model_inference": False, "inference_api": False,
              "all_declared_mechanisms_observed": all(r.get("expected_behavior_observed", True) for r in rows),
              "scope": "Synthetic real-HAPI engineering/boundary checks, not independent clinical patients or LLM performance"}
    write_json(a.out / "complete.json", report)


if __name__ == "__main__":
    main()
