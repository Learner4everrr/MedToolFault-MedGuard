"""Real-HAPI engineering gates, not clinical or LLM evidence."""
import argparse
import json
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.audit import apply_diff
from medtoolbench.research.contracts import grade, service
from medtoolbench.research.fhir import HapiProcess, normalized_state
from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.transport import CONDITIONS, CallBudget, FactorClient, StressProxy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError("Never overwrite validation evidence")
    a.out.mkdir(parents=True)
    patient = {"resourceType": "Patient", "id": "supplement-technical-patient"}
    pid = patient["id"]
    obs = {"resourceType": "Observation", "id": "lab-a", "status": "final", "subject": {"reference": "Patient/" + pid},
        "code": {"coding": [{"code": "K"}]}, "effectiveDateTime": "2023-11-13T09:00:00Z", "valueQuantity": {"value": 3, "unit": "mmol/L"}}
    task = {"patient": pid, "read_code": "K", "contract": {"answer": {"completed": True}, "required_creations": [service(pid, "4548-4")]},
        "hazards": ["clean", "transient_503", "stale", "patient_mismatch", "provenance_conflict", "commit_timeout", "no_commit_timeout"],
        "stale_resource": {**obs, "id": "old", "valueQuantity": {"value": 4, "unit": "mmol/L"}},
        "distractor_resources": [{**obs, "id": "wrong", "subject": {"reference": "Patient/other"}}]}
    checks = []
    with HapiProcess(a.image, a.out / "hapi", heap="2g") as store:
        for hazard, stress in [("clean", {}), ("commit_timeout", {}), ("no_commit_timeout", {}),
                ("transient_503+stale", {"repeat": 3}), ("transient_503", {"placement": "agent_target"}),
                ("commit_timeout", {"visibility_delay": 1}), ("no_commit_timeout", {"concurrent": True})]:
            for condition in CONDITIONS:
                before = store.reload([patient, obs])
                proxy = StressProxy(store, task, hazard, stress)
                budget = CallBudget(proxy.call)
                client = FactorClient(budget, condition)
                read = client.call("GET", f"/Observation?patient={pid}&code=K&_count=1")
                write = client.call("POST", "/ServiceRequest", service(pid, "4548-4"))
                after = store.snapshot()
                g = grade(task, before, after, {"completed": True})
                assert normalized_state(apply_diff(before, g["diff"])) == normalized_state(after)
                assert proxy.triggered == sum(bool(t["injected"]) for t in proxy.trace)
                if hazard == "commit_timeout" and not stress and condition == "medguard":
                    assert g["success"]
                if hazard == "no_commit_timeout" and not stress and condition == "medguard":
                    assert g["success"]
                checks.append({"hazard": hazard, "stress": stress, "condition": condition,
                    "state_replay": True, "calls": len(proxy.trace), "triggers": proxy.triggered,
                    "read_status": read["status"], "write_status": write["status"], "grade": g})
    write_json(a.out / "complete.json", {"all_pass": True, "checks": len(checks), "model_inference": False,
        "scope": "Synthetic real-HAPI engineering validation, not clinical results", "results": checks,
        "transport_hash": digest(Path(__file__).resolve().parents[1].joinpath("medtoolbench/supplement/transport.py").read_text())})
    print(json.dumps({"all_pass": True, "checks": len(checks)}))


if __name__ == "__main__":
    main()
