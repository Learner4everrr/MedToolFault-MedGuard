"""Freeze an additive memory execution revision; never rewrite scientific plans."""
import argparse
import json
import time
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.job_state import write_json
from medtoolbench.supplement.pipeline import load

p = argparse.ArgumentParser()
p.add_argument("--pipeline", type=Path, default=Path("data/supplement_pipeline_8gpu_v2/pipeline.json"))
p.add_argument("--out", type=Path, default=Path("data/supplement_memory_recovery_v1"))
a = p.parse_args()
base = load(a.pipeline.resolve())
if a.out.exists():
    raise RuntimeError("Never overwrite a memory execution revision")
root = Path(__file__).resolve().parents[2]
files = [root / "repo/medtoolbench/supplement" / name for name in
         ("memory_adapter.py", "memory_runner.py", "memory_execution.py", "memory_preflight.py")]
files += [Path(__file__).resolve()]
manifest = {"created_at": time.time(), "canonical_pipeline": str(a.pipeline.resolve()),
    "canonical_pipeline_hash": base["pipeline_hash"], "campaigns": base["campaigns"], "plan_hashes": base["plan_hashes"],
    "source_hashes": {str(f): digest(f.read_text()) for f in files}, "chunk_tokens": 512,
    "extra_infrastructure_attempts": 2, "execution_order": [0, 1, 3, 2], "inference_api": False,
    "session": "medtoolbench-memory-recovery-v1", "same_scientific_plans": True,
    "scope": "Gemma FFN tiling only; original batch, dtype, attention, context, budgets, seeds, tasks and grading preserved",
    "repair_policy": "Only terminal CUDA-OOM failures without audits qualify for extra whole-shard attempts; preserve all originals. Never select by accuracy.",
    "handoff_policy": "Reserve unstarted job locks; adopt GPUs only after original worker reports waiting. Never signal active model jobs.",
    "runtime_cost_policy": "Disclose tiled implementation in metadata; do not directly pool wall-time efficiency estimates across implementations.",
    "qualification": "Exact development greedy-generation comparisons, BF16 FFN numerical test, two full-context/full-decoding synthetic probes, then all A Gemma development shards independently rerun/audited in a separate non-test directory before adapted held-out inference"}
manifest["manifest_hash"] = digest(manifest)
a.out.mkdir(parents=True)
write_json(a.out / "manifest.json", manifest)
for file in files:
    destination = a.out / "frozen_source" / file.relative_to(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(file.read_bytes())
print(json.dumps({"manifest": str((a.out / "manifest.json").resolve()), "manifest_hash": manifest["manifest_hash"]}))
