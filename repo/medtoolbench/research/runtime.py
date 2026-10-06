"""Versioning, GPU leases and audited whole-batch recovery for local runs."""
import fcntl
import json
import os
import shutil
import socket
from pathlib import Path

from medtoolbench.storage import digest

CORE = ("contracts", "fhir", "models", "protocol", "transport", "run", "oracle",
        "input_validation", "runtime", "build", "job_state")


def core_fingerprint():
    package = Path(__file__).parent
    files = [package / (n + ".py") for n in CORE]
    files += [package.parent / (n + ".py") for n in ("storage", "grading", "run_local")]
    return {str(p.relative_to(package.parent)): digest(p.read_text()) for p in files}


class GpuLease:
    def __init__(self):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible not in {"0", "1"}:
            raise RuntimeError("Declare exactly one allocated GPU using CUDA_VISIBLE_DEVICES=0 or 1")
        identity = digest([str(Path(__file__).resolve().parents[3]), socket.gethostname()])[:16]
        self.path = Path("/tmp") / f"medtoolbench-{identity}-gpu{visible}.lock"
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError("Another owned run holds this GPU lease") from None
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()


def batch_keys(batch):
    return [f"{t['id']}--{h}--{c}" for t, h, c in batch]


def recover_batches(source, out, benchmark, config, batches):
    """Preserve complete audited batches; rerun incomplete batches on fresh HAPI.

    A new attempt always has a new output directory and a new server. Keeping
    whole batches preserves padded-generation peers and deterministic ordering.
    """
    from .audit import audit
    from .build import load_gzip
    source = source.resolve()
    old = json.loads((source / "config.json").read_text())
    for key in ("manifest_hash", "model", "split", "seed", "conditions", "hazards", "batch_size",
                "max_turns", "max_new_tokens", "max_context", "thinking", "expected_keys", "core_hashes"):
        if old.get(key) != config.get(key):
            raise RuntimeError("Resume configuration/code differs: " + key)
    proof = audit(source, benchmark)
    if proof["errors"] or proof["unexpected"]:
        raise RuntimeError("Cannot resume unaudited or corrupt partial evidence")
    summaries, remaining, copied = [], [], []
    for batch in batches:
        keys = batch_keys(batch)
        if not all((source / "episodes" / (k + ".json.gz")).is_file() for k in keys):
            remaining.append(batch)
            continue
        for key in keys:
            episode = source / "episodes" / (key + ".json.gz")
            row = load_gzip(episode)
            ref = Path(row["before_ref"])
            target = out / ref
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source / ref, target)
            shutil.copy2(episode, out / "episodes" / episode.name)
            summaries.append({"key": key, "success": row["grade"]["success"], "unsafe": row["grade"]["unsafe"],
                              "triggered": row["triggered"], "stop": row["stop"]})
            copied.append(key)
    return summaries, remaining, {"source": str(source), "recovered_episode_keys": copied,
                                  "partial_batch_policy": "rerun entire incomplete batch from fresh server state"}
