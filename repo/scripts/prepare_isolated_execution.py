"""Freeze a new execution addendum, preserving every scientific/source hash."""
import argparse
import json
import time
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.pipeline import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", type=Path, default=Path("data/supplement_pipeline_8gpu_v2/pipeline.json"))
    p.add_argument("--impact", type=Path, default=Path("data/input_guard_impact_v1/complete.json"))
    p.add_argument("--out", type=Path, default=Path("data/supplement_isolated_execution_v1"))
    a = p.parse_args()
    base = load(a.pipeline.resolve())
    impact = json.loads(a.impact.read_text())
    if not impact["all_scanned"] or not impact["behavior_unchanged_for_saved_trajectories"]:
        raise RuntimeError("Historical behavioral impact requires explicit resolution")
    if a.out.exists():
        raise RuntimeError("Never overwrite an execution revision")
    root = Path(__file__).resolve().parents[2]
    files = [root / "repo/medtoolbench/supplement" / name for name in (
        "input_guard.py", "serial_model.py", "isolated_manifest.py", "isolated_runner.py",
        "isolated_preflight.py", "isolated_execution.py", "memory_adapter.py", "memory_execution.py")]
    files += [root / "repo/scripts" / name for name in (
        "prepare_isolated_execution.py", "handoff_isolated_execution.py", "audit_meta_guard_impact.py",
        "validate_isolated_execution.py")]
    files += [root / "repo/tests/test_isolated_execution.py"]
    old_root = root / "data/supplement_pipeline_8gpu_v1"
    manifest = {"created_at": time.time(), "canonical_pipeline": str(a.pipeline.resolve()),
        "canonical_pipeline_hash": base["pipeline_hash"], "campaigns": base["campaigns"],
        "plan_hashes": base["plan_hashes"], "source_hashes": {str(f): digest(f.read_text()) for f in files},
        "session": "medtoolbench-isolated-execution-v1", "inference_api": False,
        "gemma_microbatch": 1, "chunk_tokens": 512, "attempts_per_job_this_revision": 1,
        "input_impact_audit": str(a.impact.resolve()), "input_impact_hash": digest(impact),
        "legacy_worker_lock_roots": [str(old_root), str(a.pipeline.resolve().parent)],
        "legacy_controllers": [
            {"module": "medtoolbench.supplement.resource_pipeline", "argument": "--pipeline",
             "path": str(old_root / "pipeline.json"), "expected_processes": 8},
            {"module": "medtoolbench.supplement.c_repair_pipeline", "argument": "--pipeline",
             "path": str(a.pipeline.resolve()), "expected_processes": 8},
            {"module": "medtoolbench.supplement.memory_execution", "argument": "--manifest",
             "path": str(root / "data/supplement_memory_recovery_v1/manifest.json"), "expected_processes": 1}],
        "failure_policy": "One outcome-independent new whole-shard attempt for unstarted or terminal OOM/meta-null jobs. On new runtime failure quarantine the shard and circuit-break its model, continuing other models/campaigns. Hash or audit failures halt dispatch. All missing assignments remain pending.",
        "gate_policy": "Shared campaign oracle plus that model's entire qualification set; independent model families/campaigns need not wait. Gemma additionally needs both GPU preflights and all 112 A shadow development episodes.",
        "execution_policy": "Six lanes prioritize healthy models; two lanes first validate Gemma, then execute its pending jobs. Idle lanes may borrow eligible work. Round-robin campaigns; qualifications first. No selection on accuracy.",
        "input_policy": "Reject top-level meta values that are not JSON objects at the shared action parser. No payload correction, no raw call, one consumed turn, normal 400 feedback. Original graders and frozen sources unchanged.",
        "numerical_policy": "Gemma physical microbatch 1 with FFN tile 512; logical episode batches, task order, precision, context, output budgets and seeds unchanged. Sampling RNG consumption and floating rounding can differ; no universal trajectory equivalence claim. Validated greedy comparisons are limited to their tested prompts.",
        "runtime_cost_policy": "Record original vs tiled vs serial execution strata; do not pool wall-time or throughput claims across them.",
        "partial_recovery_policy": "Keep all failed partial evidence, but restart the entire failed shard in a new attempt; no partial-row pooling or selective successful-input filtering."}
    manifest["manifest_hash"] = digest(manifest)
    a.out.mkdir(parents=True)
    write_json(a.out / "manifest.json", manifest)
    for file in files:
        target = a.out / "frozen_source" / file.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    print(json.dumps({"manifest": str((a.out / "manifest.json").resolve()), "manifest_hash": manifest["manifest_hash"]}))


if __name__ == "__main__":
    main()
