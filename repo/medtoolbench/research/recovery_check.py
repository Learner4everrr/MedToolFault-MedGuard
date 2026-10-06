"""Exercise real saved-evidence batch recovery, without another model generation."""
import argparse
import json
import shutil
from pathlib import Path

from .build import load_gzip
from .job_state import write_json
from .runtime import recover_batches


def main():
    p = argparse.ArgumentParser()
    for name in ("run", "benchmark", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Recovery-check output must be new")
    config = json.loads((a.run / "config.json").read_text())
    if not (a.run / "complete.json").exists():
        raise RuntimeError("Use a completed real-evidence source run")
    tasks = {t["id"]: t for t in json.loads((a.benchmark / "tasks.json").read_text())}
    keys = config["expected_keys"]
    size = config["batch_size"]
    if len(keys) < size * 2:
        raise RuntimeError("Need at least two original batches")
    source, recovered = a.out / "simulated_interrupted", a.out / "recovered"
    for folder in (source, recovered):
        (folder / "episodes").mkdir(parents=True)
        (folder / "states").mkdir()
    # Preserve one complete batch and one member of the next batch. No model,
    # server or genuine run is interrupted and no source artifact is changed.
    write_json(source / "config.json", config)
    for key in keys[:size + 1]:
        origin = a.run / "episodes" / (key + ".json.gz")
        row = load_gzip(origin)
        shutil.copy2(origin, source / "episodes" / origin.name)
        shutil.copy2(a.run / row["before_ref"], source / row["before_ref"])
    jobs = [(tasks[t], h, c) for t, h, c in (k.split("--") for k in keys)]
    batches = [jobs[i:i + size] for i in range(0, len(jobs), size)]
    summaries, remaining, proof = recover_batches(source, recovered, a.benchmark, config, batches)
    copied = proof["recovered_episode_keys"]
    exact_bytes = all((recovered / "episodes" / (k + ".json.gz")).read_bytes() ==
                      (a.run / "episodes" / (k + ".json.gz")).read_bytes() for k in copied)
    correct = copied == keys[:size] and len(summaries) == size and remaining == batches[1:] and exact_bytes
    report = {"all_pass": correct, "source_run": str(a.run.resolve()), "simulated_partial_episodes": size + 1,
              "recovered_episodes": len(summaries), "requeued_batches": len(remaining),
              "byte_identical_evidence": exact_bytes, "partial_batch_entirely_requeued": remaining == batches[1:],
              "real_independent_audit": json.loads((source / "audit.json").read_text()),
              "model_inference": False, "counts_toward_benchmark": False}
    write_json(a.out / "complete.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "real_independent_audit"}), flush=True)
    if not correct:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
