"""Freeze a post-primary follow-up matrix before any follow-up model evaluation."""
import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import extension_hashes
from medtoolbench.research.job_state import write_json
from medtoolbench.research.runtime import core_fingerprint
from medtoolbench.supplement.transport import CONDITIONS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--technical-proof", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    primary = json.loads((root / "data/primary_study_v1/plan.json").read_text())
    proof = json.loads(a.technical_proof.read_text())
    if not proof["all_pass"] or core_fingerprint() != primary["core_hashes"]:
        raise RuntimeError("Require technical validation and unchanged primary")
    if a.out.exists():
        raise RuntimeError("Never overwrite a supplement freeze")
    tasks = json.loads((Path(primary["benchmark"]) / "tasks.json").read_text())
    selected = ("qwen35_9b", "ministral3_8b", "gemma3_12b")
    systems = [s for s in primary["systems"] if s["name"] in (*selected, "llama31_8b", "phi4_mini")]
    base = dict(seed=20260917, max_turns=12, max_new_tokens=2048, max_context=16384,
                raw_call_budget=48, temperature=0, stress={})
    jobs = []

    def add(arm, candidates, models, conditions, hazards=None, settings=None, stage="test", shard_count=8):
        entries = []
        for t in candidates:
            for h in (hazards or t["hazards"]):
                if not set(h.split("+")) <= set(t["hazards"]):
                    continue
                for c in conditions:
                    entries.append({"task": t["id"], "hazard": h, "condition": c, "key": f"{t['id']}--{h}--{c}"})
        actual_settings = copy.deepcopy(base)
        actual_settings.update(settings or {})
        for model in models:
            for shard in range(shard_count):
                rows = entries[shard::shard_count]
                if rows:
                    jobs.append({"id": f"{arm}--{model}--{shard:02d}", "arm": arm, "model": model,
                        "stage": stage, "settings": actual_settings, "episodes": rows})

    dev = [next(t for t in tasks if t["split"] == "dev" and t["template"] == i) for i in (3, 4, 8, 9)]
    test = [t for t in tasks if t["split"] == "test"]
    add("qualification", dev, selected, CONDITIONS, stage="qualification", shard_count=2)
    # Technical checks, not a minimum-accuracy criterion. The two adverse-outcome
    # models use the same already-qualified original tokenizer and frozen scaffold.
    add("qualification_adverse", dev[:1], ("llama31_8b", "phi4_mini"), ("vanilla", "medguard"), stage="qualification", shard_count=1)
    add("matched_controls", test, selected, CONDITIONS)
    add("operation_matched", test, selected, ("vanilla", "medguard"), ["transient_503"], {"stress": {"placement": "agent_target"}})
    add("repeated_faults", test, selected, ("vanilla", "medguard"),
        ["transient_503", "stale", "patient_mismatch", "provenance_conflict", "commit_timeout", "no_commit_timeout"], {"stress": {"repeat": 3}})
    add("compound_faults", test, selected, ("vanilla", "medguard"),
        ["transient_503+stale", "patient_mismatch+commit_timeout", "provenance_conflict+no_commit_timeout"], {"stress": {"repeat": 1}})
    action = [t for t in test if t["contract"]["required_creations"]]
    add("visibility_delay", action, selected, ("vanilla", "medguard"), ["commit_timeout"], {"stress": {"visibility_delay": 1}})
    add("concurrent_writer", action, selected, ("vanilla", "medguard"), ["commit_timeout", "no_commit_timeout"], {"stress": {"concurrent": True}})
    add("adverse_model_actions", action, ("llama31_8b", "phi4_mini"), ("vanilla", "medguard"))
    # Hash-based, outcome-independent subset; no new claim of unseen templates.
    subset = []
    for template in range(1, 11):
        subset.extend(sorted((t for t in test if t["template"] == template), key=lambda t: digest(["supplement-subset", t["id"]]))[:5])
    for limit in (1024, 4096):
        add("tokens_" + str(limit), subset, selected, ("vanilla", "medguard"), settings={"max_new_tokens": limit}, shard_count=4)
    add("calls_24", subset, selected, ("vanilla", "medguard"), settings={"raw_call_budget": 24}, shard_count=4)
    for seed in (20260918, 20260919, 20260920):
        add("sampling_" + str(seed), subset, selected, ("vanilla", "medguard"), settings={"temperature": .7, "seed": seed}, shard_count=4)
    inputs = [Path(primary["benchmark"]) / "tasks.json", a.technical_proof.resolve()]
    source_files = [Path(__file__).resolve()]
    source_files += [root / "repo/medtoolbench/supplement" / (n + ".py") for n in ("transport", "runner", "campaign", "oracle")]
    source_files += [root / "repo/medtoolbench/research" / (n + ".py") for n in ("cluster_execution", "cluster_supervisor", "audit", "supervisor")]
    plan = {"version": 1, "project": str(root), "session": "medtoolbench-supplement-a1", "created_at": time.time(),
        "python": str(root / "env_modern/bin/python"), "benchmark": primary["benchmark"], "image": primary["image"],
        "primary_plan_hash": primary["plan_hash"], "primary_core_hashes": primary["core_hashes"],
        "source_hashes": {str(p): digest(p.read_text()) for p in source_files},
        "input_hashes": {str(p): digest(p.read_bytes().hex()) for p in inputs},
        "workers": json.loads((root / "data/primary_study_v1/execution_sixgpu_v1/deployment.json").read_text())["workers"],
        "systems": systems, "tasks": tasks, "jobs": jobs, "inference_api": False, "require_oracle": True,
        "selection": "Three capable distinct families selected after observing primary results; adverse Llama/Phi action arms retained separately. Not an independent blinded confirmatory selection.",
        "statistics": {"bootstrap": "patient/group clustered paired bootstrap", "draws": 10000,
            "primary_supplement_endpoint": "all-assigned task-weighted safe-success difference; action-required unsafe separately",
            "comparisons": "within same arm/settings; compare every control to vanilla and full guard; no multiplicity-adjusted significance claims",
            "sampling": "three prespecified seeds; aggregate within task before group resampling",
            "boundaries": "concurrent external writes are separately attributed; do not label all resulting unsafe state as agent-caused harm"},
        "remaining_separate_arms": ["external model evaluation", "semisynthetic decision/action expansion", "preexisting order variants"],
        "qualification": "All development episode keys, real-server state replay, token/call accounting must pass; never gate on accuracy"}
    plan["plan_hash"] = digest(plan)
    a.out.mkdir(parents=True)
    (a.out / "sources").mkdir()
    for name in source_files:
        # Preserve exact frozen source for recovery and later auditing.
        (a.out / "sources" / (digest(str(name))[:10] + "-" + name.name)).write_bytes(name.read_bytes())
    write_json(a.out / "plan.json", plan)
    counts = Counter()
    for j in jobs:
        counts[j["arm"]] += len(j["episodes"])
    write_json(a.out / "matrix.json", {"arms": counts, "jobs": len(jobs), "episodes": sum(counts.values()), "plan_hash": plan["plan_hash"]})
    print(json.dumps({"arms": counts, "jobs": len(jobs), "episodes": sum(counts.values()), "plan_hash": plan["plan_hash"]}, indent=2))


if __name__ == "__main__":
    main()
