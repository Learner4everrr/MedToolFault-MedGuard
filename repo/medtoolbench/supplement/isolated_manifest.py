"""Checks for the additive, fault-isolated execution revision."""
import json
from pathlib import Path

from medtoolbench.storage import digest
from .pipeline import load as load_pipeline


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def load(path):
    manifest = read(path)
    checksum = manifest.pop("manifest_hash")
    if digest(manifest) != checksum:
        raise RuntimeError("Isolated execution manifest changed")
    manifest["manifest_hash"] = checksum
    for name, value in manifest["source_hashes"].items():
        if digest(Path(name).read_text()) != value:
            raise RuntimeError("Isolated execution source changed: " + name)
    base = load_pipeline(Path(manifest["canonical_pipeline"]))
    for field in ("campaigns", "plan_hashes"):
        if manifest[field] != base[field]:
            raise RuntimeError("Canonical scientific plans changed")
    if manifest["canonical_pipeline_hash"] != base["pipeline_hash"]:
        raise RuntimeError("Canonical pipeline changed")
    if manifest["gemma_microbatch"] != 1 or manifest["chunk_tokens"] != 512:
        raise RuntimeError("Undeclared model execution configuration")
    if manifest["attempts_per_job_this_revision"] != 1:
        raise RuntimeError("Undeclared retry policy")
    return manifest


def proof_passes(path, manifest):
    record = read(path)
    return bool(record and record.get("all_pass") and record.get("manifest_hash") == manifest["manifest_hash"])


def preflight_passes(root, manifest):
    return all(proof_passes(Path(root) / "preflight" / mode / "complete.json", manifest)
               for mode in ("compare", "stress"))


def require_cpu_validation(root, manifest):
    if not proof_passes(Path(root) / "validation/unit.json", manifest):
        raise RuntimeError("CPU regression suite is not validated for this revision")
    if not proof_passes(Path(root) / "validation/hapi.json", manifest):
        raise RuntimeError("Real-HAPI input regression is not validated for this revision")
    impact = read(manifest["input_impact_audit"])
    if not impact or not impact["all_scanned"] or digest(impact) != manifest["input_impact_hash"]:
        raise RuntimeError("Historical input-impact audit missing or changed")
    return impact
