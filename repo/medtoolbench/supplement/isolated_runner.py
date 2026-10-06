"""Versioned parser/serial-inference substitutions to unchanged frozen runners."""
import argparse
import importlib
import sys
from pathlib import Path

from .input_guard import install as install_input_guard
from .isolated_manifest import load, preflight_passes, proof_passes, require_cpu_validation
from .runner import SampleModel, load_plan
from .serial_model import SerialGemma


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--execution-manifest", type=Path, required=True)
    p.add_argument("--shadow-development", action="store_true")
    a, remaining = p.parse_known_args()
    manifest = load(a.execution_manifest)
    control = a.execution_manifest.parent
    require_cpu_validation(control, manifest)
    path = Path(remaining[remaining.index("--plan") + 1]).resolve()
    if str(path) not in manifest["campaigns"]:
        raise RuntimeError("Plan is outside execution scope")
    plan = load_plan(path)
    job = next(j for j in plan["jobs"] if j["id"] == remaining[remaining.index("--job") + 1])
    if a.shadow_development and (job["stage"] != "qualification" or job["model"] != "gemma3_12b"):
        raise RuntimeError("Shadow regression cannot select test episodes")
    if job["model"] == "gemma3_12b":
        if not preflight_passes(control, manifest):
            raise RuntimeError("Both Gemma stress and comparison gates must pass")
        if not a.shadow_development and not proof_passes(control / "gemma_development/complete.json", manifest):
            raise RuntimeError("All A Gemma shadow development jobs must pass")
    guard = install_input_guard()
    record = {"manifest": str(a.execution_manifest.resolve()), "manifest_hash": manifest["manifest_hash"],
        "scientific_plan_hash": plan["plan_hash"], "source_hashes": manifest["source_hashes"],
        "input_guard": guard, "shadow_development": a.shadow_development,
        "physical_microbatch": 1 if job["model"] == "gemma3_12b" else "original",
        "scientific_budgets_labels_grader_unchanged": True}
    parent = SerialGemma if job["model"] == "gemma3_12b" else SampleModel

    class RecordedModel(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.metadata["isolated_execution"] = record

    from . import runner
    runner.SampleModel = RecordedModel
    module = importlib.import_module(plan["runner_module"])
    if plan["runner_module"].endswith("external_runner"):
        module.SampleModel = RecordedModel
    original_write = runner.write_json

    def recorded_write(path, value):
        if Path(path).name == "config.json":
            value = {**value, "isolated_execution": record}
        return original_write(path, value)

    runner.write_json = recorded_write
    if plan["runner_module"].endswith("external_runner"):
        module.write_json = recorded_write
    sys.argv = [str(Path(module.__file__)), *remaining]
    module.main()


if __name__ == "__main__":
    main()
