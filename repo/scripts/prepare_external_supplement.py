"""Freeze original external questions/labels with a local complete-history SDK."""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.build import load_gzip
from medtoolbench.research.job_state import write_json
from medtoolbench.research.runtime import core_fingerprint
from medtoolbench.supplement.external_tools import KINDS, ExternalTools, Retrieval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    primary = json.loads((root / "data/primary_study_v1/plan.json").read_text())
    verified = root / "data/external_validation_all_v1"
    if not json.loads((verified / "complete.json").read_text())["all_pass"]:
        raise RuntimeError("Full external reference validation required")
    if a.out.exists():
        raise RuntimeError("Never overwrite an external freeze")
    questions = [q for q in load_gzip(root / "data/external_inventory_v2/questions.json.gz") if q["main_table_name"] in {"chartevents", "labevents"}]
    patients = sorted({q["patient_fhir_id"] for q in questions})
    sources, proof = {}, []
    hazards = ("clean", "interface_drift", "transient_503", "partial_history", "patient_mismatch", "provenance_conflict")
    for i, pid in enumerate(patients):
        path = verified / "patients" / pid / "http_resources.json.gz"
        resources = load_gzip(path)
        other = patients[(i + 1) % len(patients)]
        other_rows = load_gzip(verified / "patients" / other / "http_resources.json.gz")
        data = {k: [r for r in resources if r["resourceType"] == k] for k in KINDS}
        alternate = {k: [r for r in other_rows if r["resourceType"] == k] for k in KINDS}
        sources[pid] = {"path": str(path), "hash": digest(resources), "other_patient": other, "resources": len(resources)}
        for hazard in hazards:
            tools = ExternalTools(Retrieval(data, pid, hazard, alternate), "medguard")
            for kind in KINDS:
                result = tools.call({"tool": "get_resources", "resource_type": kind, "patient_fhir_id": pid})
                # A guard that drops genuine source rows would invalidate the
                # external projection; do not silently change reference labels.
                if result["status"] != 200 or tools.retrieved[kind] != data[kind]:
                    raise RuntimeError("External source recovery mismatch: " + pid + " " + hazard + " " + kind)
            proof.append({"patient": pid, "hazard": hazard, "all_source_rows_recovered": True})
        print(json.dumps({"patients_prepared": i + 1, "total": len(patients)}), flush=True)
    systems = [s for s in primary["systems"] if s["name"] in {"qwen35_9b", "ministral3_8b", "gemma3_12b"}]
    jobs = []
    def add(stage, candidates):
        grouped = defaultdict(list)
        for q in candidates:
            grouped[q["patient_fhir_id"]].append(q)
        blocks = []
        for pid, qs in sorted(grouped.items()):
            qs = sorted(qs, key=lambda q: q["question_id"])
            for offset in range(0, len(qs), 8):
                blocks.append(qs[offset:offset + 8])
        # Put development jobs from all three families first, six short shards.
        for system in systems:
            for i, block in enumerate(blocks):
                entries = [{"question": q["question_id"], "hazard": h, "condition": c,
                    "key": q["question_id"] + "--" + h + "--" + c}
                    for q in block for h in hazards for c in ("vanilla", "informed", "medguard")]
                jobs.append({"id": stage + "--" + system["name"] + "--" + str(i).zfill(3), "arm": "external_qa",
                    "stage": stage, "model": system["name"], "episodes": entries,
                    "settings": {"max_turns": 20, "max_new_tokens": 2048, "max_context": 16384, "raw_call_budget": 48, "seed": 20260917}})
    dev_patients = sorted({q["patient_fhir_id"] for q in questions if q["patient_group_split"] == "dev"})[:2]
    dev = []
    for pid in dev_patients:
        dev.extend(sorted((q for q in questions if q["patient_fhir_id"] == pid), key=lambda q: q["question_id"])[:2])
    add("qualification", dev)
    add("test", [q for q in questions if q["patient_group_split"] == "test"])
    sources_code = [Path(__file__).resolve()]
    sources_code += [root / "repo/medtoolbench/supplement" / (n + ".py") for n in ("runner", "campaign", "external_runner", "external_tools", "code_worker")]
    sources_code += [root / "repo/medtoolbench/research" / (n + ".py") for n in ("external_store", "cluster_execution", "cluster_supervisor", "sql_reference")]
    plan = {"version": 1, "project": str(root), "session": "medtoolbench-supplement-b1", "created_at": time.time(),
        "python": str(root / "env_modern/bin/python"), "image": primary["image"], "inference_api": False,
        "runner_module": "medtoolbench.supplement.external_runner", "require_oracle": False,
        "primary_core_hashes": core_fingerprint(), "source_hashes": {str(p): digest(p.read_text()) for p in sources_code},
        "input_hashes": {str(verified / "complete.json"): digest((verified / "complete.json").read_bytes().hex())},
        "workers": json.loads((root / "data/primary_study_v1/execution_sixgpu_v1/deployment.json").read_text())["workers"],
        "systems": systems, "questions": questions, "sources": sources, "jobs": jobs,
        "scope": "Original FHIR-AgentBench questions/labels; local read-only full-history retrieval plus restricted Python calculation; not EHR write safety",
        "adaptations": ["local real HAPI replaces cloud FHIR", "complete receipt-based SDK retrieval to avoid search caps",
            "bounded Python subset replaces unrestricted Python tool", "exact numeric/tabular reference grading, no model judge"],
        "statistics": {"bootstrap_draws": 10000, "cluster": "patient", "weighting": "average assigned hazards within question",
            "endpoint": "paired answer-correctness change", "conditional_endpoint": "secondary, eligibility fixed by vanilla-clean",
            "no_claims": ["patient harm", "write safety", "independently sampled hospital cohort", "fault prevalence", "public preregistration"]}}
    plan["plan_hash"] = digest(plan)
    a.out.mkdir(parents=True)
    (a.out / "sources").mkdir()
    for path in sources_code:
        (a.out / "sources" / (digest(str(path))[:10] + "-" + path.name)).write_bytes(path.read_bytes())
    write_json(a.out / "plan.json", plan)
    write_json(a.out / "construction_audit.json", {"all_pass": True, "checks": len(proof), "results": proof,
        "reference_answer_audit": str(verified / "complete.json"), "model_inference": False})
    write_json(a.out / "matrix.json", {"jobs": len(jobs), "episodes": sum(len(j["episodes"]) for j in jobs),
        "test_questions": sum(q["patient_group_split"] == "test" for q in questions), "qualification_questions": len(dev),
        "plan_hash": plan["plan_hash"]})
    print((a.out / "matrix.json").read_text())


if __name__ == "__main__":
    main()
