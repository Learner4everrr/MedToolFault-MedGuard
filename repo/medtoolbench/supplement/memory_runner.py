"""Explicit, separately hashed execution addendum to unchanged scientific plans."""
import argparse
import importlib
import json
import sys
from pathlib import Path

from medtoolbench.research.job_state import write_json
from .memory_adapter import install
from .runner import SampleModel, load_plan


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--memory-manifest", type=Path, required=True)
    parser.add_argument("--development-validation", action="store_true")
    args, remaining = parser.parse_known_args()
    from .memory_execution import load
    manifest = load(args.memory_manifest)
    path = Path(remaining[remaining.index("--plan") + 1]).resolve()
    if str(path) not in manifest["campaigns"]:
        raise RuntimeError("Plan not covered by execution addendum")
    plan = load_plan(path)
    proof = json.loads((args.memory_manifest.parent / "preflight/complete.json").read_text())
    if not proof["all_pass"] or proof["manifest_hash"] != manifest["manifest_hash"]:
        raise RuntimeError("Memory regression gate did not pass")
    job_id = remaining[remaining.index("--job") + 1]
    job = next(j for j in plan["jobs"] if j["id"] == job_id)
    if job["model"] != "gemma3_12b":
        raise RuntimeError("Memory adapter is scoped to Gemma only")
    if args.development_validation:
        if job["stage"] != "qualification":
            raise RuntimeError("Development-only gate cannot run held-out episodes")
    else:
        development = json.loads((args.memory_manifest.parent / "qualification/complete.json").read_text())
        if not development["all_pass"] or development["manifest_hash"] != manifest["manifest_hash"]:
            raise RuntimeError("Full Gemma development regression must pass first")
    record = {"manifest": str(args.memory_manifest.resolve()), "manifest_hash": manifest["manifest_hash"],
              "source_hashes": manifest["source_hashes"], "scientific_plan_hash": plan["plan_hash"],
              "preflight": str(args.memory_manifest.parent / "preflight/complete.json")}
    record["development_validation_only"] = args.development_validation

    class TiledModel(SampleModel):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if self.kind != "gemma3":
                raise RuntimeError("Unexpected model architecture")
            self.metadata["memory_execution"] = {**record, **install(self.model, manifest["chunk_tokens"])}

    # Process-local substitution only. Baseline frozen files and source hashes
    # remain intact, and config/model records explicitly disclose the addendum.
    from . import runner
    runner.SampleModel = TiledModel
    module = importlib.import_module(plan["runner_module"])
    if plan["runner_module"].endswith("external_runner"):
        module.SampleModel = TiledModel
    original_write = runner.write_json

    def recorded_write(path, value):
        if Path(path).name == "config.json":
            value = {**value, "memory_execution": record}
        return original_write(path, value)

    runner.write_json = recorded_write
    if plan["runner_module"].endswith("external_runner"):
        module.write_json = recorded_write
    sys.argv = [str(Path(module.__file__)), *remaining]
    module.main()


if __name__ == "__main__":
    main()
