"""Freeze the entire primary held-out matrix after explicit qualification gates."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from medtoolbench.storage import digest
from .audit import audit
from .job_state import write_json
from .runtime import core_fingerprint

CONDITIONS = ["vanilla", "warning", "retry_only", "medguard", "oracle_hint"]


def matrix(tasks, systems, shards=8):
    selected = [t for t in tasks if t["split"] == "test"]
    jobs = []
    for shard in range(shards):
        subset = selected[shard::shards]
        if not subset:
            continue
        for system in systems:
            jobs.append({"id": f"{system['name']}--shard{shard:02d}", "model": system["name"],
                         "task_shard": shard, "task_shards": shards,
                         "task_ids": [t["id"] for t in subset],
                         "expected_episodes": sum(len(t["hazards"]) * len(CONDITIONS) for t in subset)})
    return jobs


def freeze(benchmark, validation, registry, qualifications, image, output):
    if output.exists():
        raise RuntimeError("Never overwrite a frozen study")
    tasks = json.loads((benchmark / "tasks.json").read_text())
    fingerprint = core_fingerprint()
    vconfig = json.loads((validation / "config.json").read_text())
    vcomplete = json.loads((validation / "complete.json").read_text())
    vaudit = json.loads((validation / "audit.json").read_text())
    if vconfig.get("core_hashes") != fingerprint or vconfig["task_manifest_hash"] != digest(tasks):
        raise RuntimeError("Oracle validation does not match the current source and dataset")
    if not vcomplete["all_valid"] or not vaudit["all_pass"] or set(vconfig["selected_task_ids"]) != {t["id"] for t in tasks}:
        raise RuntimeError("Full independently audited oracle validation is required")
    systems = json.loads(registry.read_text())["systems"]
    if len(systems) != 9 or len({s["family"] for s in systems}) != 5:
        raise RuntimeError("Unexpected system/family matrix")
    if set(qualifications) != {s["name"] for s in systems}:
        raise RuntimeError("Every planned system must be development-qualified")
    proofs = {}
    for system in systems:
        run = Path(qualifications[system["name"]]).resolve()
        config = json.loads((run / "config.json").read_text())
        if config["split"] != "dev" or Path(config["model"]).resolve() != Path(system["path"]).resolve():
            raise RuntimeError("Wrong model/split in qualification")
        if config["max_new_tokens"] != 2048 or config["thinking"] != system["thinking"]:
            raise RuntimeError("Qualification must use the declared reasoning/token profile")
        proof = audit(run, benchmark)
        if not proof["all_pass"]:
            raise RuntimeError("Incomplete development qualification: " + system["name"])
        if system["name"] in {"qwen35_9b", "ministral3_8b"} and set(config["conditions"]) != set(CONDITIONS):
            raise RuntimeError("The two reference systems require all five development controls")
        proofs[system["name"]] = {"run": str(run), "audit": proof, "accuracy_used_for_admission": False}
    with image.open("rb") as stream:
        image_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    jobs = matrix(tasks, systems)
    if len(jobs) != 72 or sum(j["expected_episodes"] for j in jobs) != 47745:
        raise RuntimeError("Primary matrix count differs from the declared plan")
    plan = {"version": 1, "frozen_at": datetime.now(timezone.utc).isoformat(), "phase": "primary held-out",
            "benchmark": str(benchmark.resolve()), "validation": str(validation.resolve()),
            "manifest_hash": digest(tasks), "core_hashes": fingerprint, "image": str(image.resolve()), "image_sha256": image_hash,
            "analysis_hashes": {name: hashlib.sha256(Path(__file__).with_name(name + ".py").read_bytes()).hexdigest()
                                for name in ("analyze", "study_report", "audit", "validation_audit")},
            "systems": systems, "qualifications": proofs, "conditions": CONDITIONS,
            "generation": {"max_turns": 12, "max_new_tokens": 2048, "max_context": 16384, "seed": 20260911},
            "jobs": jobs, "expected_episodes": sum(j["expected_episodes"] for j in jobs), "inference_api": False,
            "failure_policy": {"max_attempts": 2, "resume": "audited complete batches only, same source/config", "accuracy_stopping": False},
            "primary_estimand": "Per-system paired MedGuard-minus-vanilla safe success on vanilla-clean-capable tasks; average hazards within task, resample patient groups",
            "secondary_estimands": ["all-task guard gain", "clean-to-fault drop", "persistent/transient unsafe mutations by action branch", "duplicate writes on both timeout controls", "tokens and actual tool calls"],
            "statistics": {"patient_cluster_bootstrap_draws": 10000, "interval": "percentile 95%", "confirmatory_p_values": False,
                           "note": "Effect-size/uncertainty study; per-hazard and cross-system exploratory contrasts do not establish multiplicity-adjusted significance"},
            "not_in_this_primary_matrix": ["external model evaluation", "severity/compound stress", "guard component ablations", "concurrent/eventual-consistency boundary studies"],
            "scope_note": "Primary matrix completion alone is not completion of the whole research project or proof of publication suitability"}
    plan["plan_hash"] = digest(plan)
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    return plan


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("benchmark", "validation", "registry", "qualifications", "image", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = freeze(a.benchmark, a.validation, a.registry, json.loads(a.qualifications.read_text()), a.image, a.out)
    print(json.dumps({k: result[k] for k in ("plan_hash", "frozen_at", "expected_episodes")}, indent=2))
