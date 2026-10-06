"""Prospective, outcome-independent amendment to the unfinished external B study.

New inference stratum: no historical model rows are pooled. Original external
questions, labels, tools, faults, parsing, budgets and graders are unchanged.
"""
import argparse
import collections
import copy
import hashlib
import json
import random
import time
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.job_state import write_json
from .runner import load_plan

ROOT = Path(__file__).resolve().parents[3]
CONTROL = ROOT / "data/b_lite_v1"
BASE = ROOT / "data/supplement_b_v2/plan.json"
LLAMA = ROOT / "data/acd_llama_expansion_v1/campaign_a/plan.json"
MODELS = ("qwen35_9b", "llama31_8b")
HAZARDS = ("clean", "interface_drift", "transient_503", "partial_history",
           "patient_mismatch", "provenance_conflict")
POLICIES = ("vanilla", "informed", "medguard")
SETTINGS = dict(max_turns=20, max_new_tokens=2048, max_context=16384,
                raw_call_budget=48, seed=20260917)
SEED = 20260927
LEGACY = (BASE, ROOT / "data/supplement_b_v1/plan.json",
          ROOT / "data/gptoss_replacement_v1/campaign_b/plan.json",
          ROOT / "data/gptoss_protocol_repair_v2/campaign_b/plan.json")


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def touched_patients(plans):
    """Conservatively exclude a whole patient if ANY old B model attempted it.

    Only existence/identity is used, never accuracy, stop reasons or OOM inputs.
    """
    touched, evidence = set(), []
    for path in plans:
        plan = read(path)
        questions = {q["question_id"]: q for q in plan["questions"]}
        for job in plan["jobs"]:
            folder = path.parent / "jobs" / job["id"]
            if job["stage"] != "test" or not folder.exists():
                continue
            attempts = [p for p in folder.glob("attempt*") if p.is_dir()]
            if attempts or (folder / "result.json").exists():
                patients = sorted({questions[e["question"]]["patient_fhir_id"]
                                   for e in job["episodes"]})
                touched.update(patients)
                evidence.append({"plan": str(path), "job": job["id"],
                                 "model": job["model"], "patients": patients})
    return touched, evidence


def select_questions(metadata, count=100, seed=SEED):
    """Cover all candidate patients/templates, then balance questions/patient.

    This function deliberately cannot receive labels or previous outcomes.
    Ties are fixed-seed hash ranks. This is a designed coverage sample, NOT a
    simple random or population-representative sample.
    """
    allowed = {"id", "patient", "template", "history_records", "history_bin"}
    if any(set(r) != allowed for r in metadata):
        raise ValueError("Selection accepts covariates only, never outcomes/labels")
    if len({r["id"] for r in metadata}) != len(metadata) or not 0 < count <= len(metadata):
        raise ValueError("Invalid sampling population/count")
    rank = lambda r: digest([seed, r["id"]])
    population = {r["id"]: r for r in metadata}
    missing_patients = {r["patient"] for r in metadata}
    missing_templates = {r["template"] for r in metadata}
    selected, patient_counts, template_counts = [], collections.Counter(), collections.Counter()
    while missing_patients or missing_templates:
        def key(r):
            gain = int(r["patient"] in missing_patients) + int(r["template"] in missing_templates)
            return (-gain, patient_counts[r["patient"]], template_counts[r["template"]], rank(r))
        chosen = min(population.values(), key=key)
        selected.append(population.pop(chosen["id"]))
        patient_counts[chosen["patient"]] += 1
        template_counts[chosen["template"]] += 1
        missing_patients.discard(chosen["patient"])
        missing_templates.discard(chosen["template"])
        if len(selected) > count:
            raise ValueError("Requested count cannot satisfy coverage")
    while len(selected) < count:
        chosen = min(population.values(), key=lambda r: (
            patient_counts[r["patient"]], template_counts[r["template"]], rank(r)))
        selected.append(population.pop(chosen["id"]))
        patient_counts[chosen["patient"]] += 1
        template_counts[chosen["template"]] += 1
    return sorted(selected, key=lambda r: r["id"])


def metadata_for(questions, sources):
    patients = sorted({q["patient_fhir_id"] for q in questions},
                      key=lambda p: (sources[p]["resources"], p))
    bins = {p: min(2, 3 * i // len(patients)) for i, p in enumerate(patients)}
    return [{"id": q["question_id"], "patient": q["patient_fhir_id"],
             "template": q["template"], "history_records": sources[q["patient_fhir_id"]]["resources"],
             "history_bin": bins[q["patient_fhir_id"]]} for q in questions]


def make_jobs(questions, stage):
    groups = collections.defaultdict(list)
    for q in questions:
        groups[q["patient_fhir_id"]].append(q)
    blocks = []
    # At most two questions / 36 episodes per independently recoverable shard.
    for patient, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda q: q["question_id"])
        for offset in range(0, len(rows), 2):
            blocks.append(rows[offset:offset + 2])
    random.Random(SEED).shuffle(blocks)
    jobs = []
    for i, block in enumerate(blocks):
        for model in MODELS:
            episodes = [{"question": q["question_id"], "hazard": h, "condition": c,
                         "key": q["question_id"] + "--" + h + "--" + c}
                        for q in block for h in HAZARDS for c in POLICIES]
            jobs.append({"id": f"{stage}--{model}--{i:03d}", "arm": "external_qa_lite",
                         "stage": stage, "model": model, "episodes": episodes,
                         "settings": dict(SETTINGS)})
    return jobs


def implementation_paths(base):
    return sorted(set(base["source_hashes"]) | {
        str(Path(__file__)), str(Path(__file__).with_name("b_lite_execution.py")),
        str(ROOT / "repo/tests/test_b_lite.py"),
        str(ROOT / "repo/medtoolbench/supplement/memory_execution.py"),
        str(ROOT / "repo/medtoolbench/supplement/report.py")})


def preserve_inventory(control):
    # Archive identifiers/hashes, never rename/delete/rewrite historical evidence.
    roots = [p.parent for p in LEGACY] + [ROOT / "data/b_qwen_priority_v1"]
    paths = []
    for root in roots:
        paths.extend(root.glob("*.json"))
        if (root / "jobs").exists():
            for pattern in ("*/result.json", "*/attempt*/config.json", "*/attempt*/audit.json",
                            "*/attempt*/failure.json", "*/attempt*/complete.json",
                            "*/attempt*/episodes/*.json.gz"):
                paths.extend((root / "jobs").glob(pattern))
    evidence = {str(p): sha(p) for p in sorted(set(paths)) if p.is_file()}
    write_json(control / "legacy_inventory.json", {"files": evidence, "time": time.time(),
        "legacy_rows_pooled": False, "old_b_status": read(ROOT / "data/b_qwen_priority_v1/status.json")})


def prepare(control):
    control = Path(control).resolve()
    if control.exists():
        raise RuntimeError("Never overwrite a B-lite amendment")
    base, llama = load_plan(BASE), load_plan(LLAMA)
    touched, exposure = touched_patients(LEGACY)
    pool = [q for q in base["questions"] if q["patient_group_split"] == "test"
            and q["patient_fhir_id"] not in touched]
    metadata = metadata_for(pool, base["sources"])
    selected = select_questions(metadata)
    selected_ids = {r["id"] for r in selected}
    test = [q for q in pool if q["question_id"] in selected_ids]
    # Four development patients: the two original qualification patients plus
    # the two largest remaining histories. No test trajectories enter gates.
    all_dev = [q for q in base["questions"] if q["patient_group_split"] == "dev"]
    dev_patients = sorted({q["patient_fhir_id"] for q in all_dev})[:2]
    remaining = sorted({q["patient_fhir_id"] for q in all_dev} - set(dev_patients),
                       key=lambda p: (-base["sources"][p]["resources"], p))
    dev_patients += remaining[:2]
    dev = [q for patient in dev_patients for q in sorted(
        (q for q in all_dev if q["patient_fhir_id"] == patient),
        key=lambda q: digest([SEED, q["question_id"]]))[:2]]
    if {q["patient_fhir_id"] for q in dev} & {q["patient_fhir_id"] for q in test}:
        raise RuntimeError("Development/test patient overlap")
    systems = [copy.deepcopy(next(s for s in (base if m == MODELS[0] else llama)["systems"]
                                  if s["name"] == m)) for m in MODELS]
    for system in systems:
        system.update(batch_size=2, status="B-lite fixed two-item batching; gated by development only")
        if not (Path(system["path"]) / "config.json").exists():
            raise RuntimeError("Missing local model: " + system["name"])
    control.mkdir(parents=True)
    preserve_inventory(control)
    selection = {"seed": SEED, "count": len(test), "pool_count": len(pool),
        "pool_patients": len({r["patient"] for r in metadata}),
        "pool_templates": len({r["template"] for r in metadata}),
        "selected": selected, "excluded_previously_attempted_patients": sorted(touched),
        "exposure_inventory": exposure, "outcomes_used_for_selection": False,
        "method": "fixed-seed coverage sample; cover patients/templates then balance patient/question counts",
        "generalization_scope": "selected questions in the not-previously-attempted external patient remainder",
        "not_random_population_sample": True, "public_preregistration": False}
    write_json(control / "selection.json", selection)
    code = implementation_paths(base)
    inputs = dict(base["input_hashes"])
    construction = ROOT / "data/supplement_b_v1/construction_audit.json"
    if not read(construction).get("all_pass"):
        raise RuntimeError("Original full-source fault-recovery construction audit is required")
    for p in (*LEGACY, LLAMA, construction, control / "selection.json", control / "legacy_inventory.json"):
        inputs[str(p)] = digest(p.read_bytes().hex())
    # Small model artifacts are immutable inputs; pinned snapshots and weight
    # sizes are retained from the earlier model inventory, without downloading.
    for system in systems:
        directory = Path(system["path"])
        for name in ("config.json", "tokenizer_config.json", "generation_config.json",
                     "tokenizer.json", "model.safetensors.index.json"):
            p = directory / name
            if p.is_file():
                inputs[str(p)] = digest(p.read_bytes().hex())
    plan = {**{k: copy.deepcopy(base[k]) for k in ("project", "python", "image", "primary_core_hashes",
                "workers", "sources", "scope", "adaptations")},
        "version": "b-lite-1", "created_at": time.time(), "session": "medtoolbench-b-lite-v1",
        "inference_api": False, "require_oracle": False,
        "runner_module": "medtoolbench.supplement.external_runner",
        "systems": systems, "questions": dev + test,
        "jobs": make_jobs(dev, "qualification") + make_jobs(test, "test"),
        "source_hashes": {p: digest(Path(p).read_text()) for p in code}, "input_hashes": inputs,
        "amendment": {"reason": "user-authorized reduced external validation after infrastructure cost/OOM",
            "historical_test_outcomes_exist": True, "old_rows_pooled": False,
            "new_fixed_batch_size": 2, "original_fixed_batch_size": 4,
            "bitwise_equivalence_claimed": False, "fast_kernels_enabled": False,
            "model_selection": "two existing local families, Qwen and Llama; not selected on B-lite outcomes"},
        "statistics": {"bootstrap_draws": 10000, "cluster": "patient", "seed": SEED,
            "weighting": "equal patients, then equal selected questions, then equal assigned nonclean hazards",
            "primary_comparisons": ["medguard-vanilla", "medguard-informed"],
            "endpoint": "paired answer correctness under assigned faults",
            "secondary": ["paired clean-minus-fault degradation", "vanilla-clean-capable subset",
                          "clean overhead", "per-hazard rates", "tokens and raw calls", "stop reasons"],
            "missing_infrastructure": "never silently drop; report assigned denominator and worst/best bounds",
            "no_confirmatory_p_values": True, "no_claims": base["statistics"]["no_claims"]},
        "execution": {"maximum_shard_questions": 2, "automatic_attempts_per_job": 1,
            "independent_failure_isolation": True, "model_circuit_breaker_failures": 3,
            "integrity_errors_stop_all_dispatch": True, "completed_shard_resume": True,
            "partial_failed_rows_pooled": False, "stress_prompt_tokens": 14336,
            "stress_output_tokens": 2048, "stress_batch_size": 2,
            "stress_max_reserved_fraction": .95, "stall_seconds": 7200, "wall_seconds": 21600}}
    plan["plan_hash"] = digest(plan)
    write_json(control / "plan.json", plan)
    (control / "sources").mkdir()
    for p in code:
        (control / "sources" / (digest(p)[:12] + "-" + Path(p).name)).write_bytes(Path(p).read_bytes())
    write_json(control / "matrix.json", {"plan_hash": plan["plan_hash"], "formal_episodes": len(test) * 36,
        "development_episodes": len(dev) * 36, "test_questions": len(test),
        "test_patients": len({q["patient_fhir_id"] for q in test}),
        "test_templates": len({q["template"] for q in test}), "models": list(MODELS),
        "selected_history_bins": dict(collections.Counter(r["history_bin"] for r in selected)),
        "jobs": len(plan["jobs"]), "fixed_batch_size": 2})
    return plan


def patient_effect(values, draws=10000, seed=SEED):
    """values maps patient -> selected-question paired differences."""
    means = [sum(v) / len(v) for _, v in sorted(values.items()) if v]
    if not means:
        return {"difference": None, "ci95": None, "patients": 0}
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(draws))
    return {"difference": sum(means) / len(means), "ci95": [samples[int(.025 * draws)],
            samples[min(draws - 1, int(.975 * draws))]], "patients": len(means), "bootstrap_draws": draws}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, default=CONTROL)
    args = parser.parse_args()
    prepare(args.control)
    print((args.control / "matrix.json").read_text())
