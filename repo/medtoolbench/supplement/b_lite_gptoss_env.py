"""Runtime-only GPT-OSS B-lite recovery; completed model evidence is immutable.

The first attempt failed its kernels dependency check before model generation.
Reuse the existing dedicated environment, not a package upgrade or BF16 fallback.
Repeat the original memory/development gates before any held-out dispatch.
"""
import copy
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import FileLease, validate_worker
from medtoolbench.research.job_state import write_json
from . import b_lite_remaining as previous
from .b_lite import ROOT, read, sha
from .runner import load_plan

PREVIOUS = ROOT / "data/b_lite_remaining_models_v1"
CONTROL = ROOT / "data/b_lite_gptoss_env_v1"
PYTHON = ROOT / "env_gptoss/bin/python"
MODULE = "medtoolbench.supplement.b_lite_gptoss_env"
MODEL = "gptoss20b"
OLD_INSTALL = previous.install
PACKAGES = ("torch", "transformers", "triton", "kernels", "huggingface_hub", "tokenizers", "accelerate")
REQUIRED = {"torch": "2.8.0", "transformers": "5.17.0", "triton": "3.4.0", "kernels": "0.16.0"}
MUTABLE_FIELDS = {"plan_hash", "version", "created_at", "session", "python", "systems", "jobs",
                  "source_hashes", "input_hashes", "scheduling", "runtime_recovery"}


def check_runtime(snapshot):
    if snapshot["prefix"] != str(ROOT / "env_gptoss"):
        raise RuntimeError("Use the existing dedicated env_gptoss environment")
    if any(snapshot["versions"].get(k) != v for k, v in REQUIRED.items()):
        raise RuntimeError("Dedicated runtime dependencies changed")
    if not snapshot["kernels_available"] or not snapshot["triton_available"]:
        raise RuntimeError("Native MXFP4 dependency checks failed; fallback forbidden")


def runtime_snapshot():
    from transformers.utils import is_kernels_available, is_triton_available
    snapshot = {"prefix": str(Path(sys.prefix).absolute()), "executable": str(Path(sys.executable).absolute()),
        "python_version": sys.version, "versions": {n: importlib.metadata.version(n) for n in PACKAGES},
        "kernels_available": is_kernels_available(), "triton_available": is_triton_available("3.4.0")}
    check_runtime(snapshot)
    return snapshot


def check_failure(result, plan_hash):
    if (result.get("state") != "quarantined" or result.get("stage") != "preflight" or
            result.get("model") != MODEL or result.get("plan_hash") != plan_hash or
            result.get("assigned_episodes") != 0 or result.get("partial_rows_counted") is not False or
            "MXFP4 dependencies invalid; BF16 fallback is forbidden" not in result.get("failure", {}).get("error", "")):
        raise RuntimeError("Only the documented pre-generation dependency failure qualifies")


def recovered_plan(old, snapshot):
    plan = copy.deepcopy(old)
    plan.pop("plan_hash")
    plan.update(version="b-lite-gptoss-env-recovery-1", created_at=time.time(),
        session="medtoolbench-b-lite-gptoss-env-v1", python=str(PYTHON),
        systems=[copy.deepcopy(s) for s in old["systems"] if s["name"] == MODEL],
        jobs=[copy.deepcopy(j) for j in old["jobs"] if j["model"] == MODEL])
    plan["scheduling"].update(model_order=[MODEL], initial_gpus=list(range(8)))
    plan["runtime_recovery"] = {
        "authorization": "User 2026-09-30: continue experiments after the dependency failure",
        "previous_control": str(PREVIOUS), "previous_plan_hash": old["plan_hash"],
        "previous_python": old["python"], "dedicated_python": str(PYTHON), "runtime": snapshot,
        "change": "Existing env_gptoss instead of env_modern; no packages upgraded",
        "failure_stage": "pre-generation dependency check; no development/formal GPT-OSS rows exist",
        "previous_failure_preserved": True, "prior_models_rerun": False, "old_rows_pooled": False,
        "model_weights_precision_prompts_parser_tools_labels_budgets_unchanged": True,
        "memory_and_complete_development_gates_required": True, "inference_api": False}
    return plan


def assert_science(plan, old):
    for key in set(old) - MUTABLE_FIELDS:
        if plan.get(key) != old[key]:
            raise RuntimeError("Runtime recovery changed scientific field: " + key)
    if plan["systems"] != [s for s in old["systems"] if s["name"] == MODEL] or len(plan["systems"]) != 1:
        raise RuntimeError("Exact prior GPT-OSS system required")
    if plan["jobs"] != [j for j in old["jobs"] if j["model"] == MODEL]:
        raise RuntimeError("Original GPT-OSS development/formal assignments changed")
    if plan["python"] != str(PYTHON):
        raise RuntimeError("Incorrect recovery interpreter")
    scheduling = copy.deepcopy(old["scheduling"])
    scheduling.update(model_order=[MODEL], initial_gpus=list(range(8)))
    if plan["scheduling"] != scheduling:
        raise RuntimeError("Resource ownership or scheduling contract changed")
    for field in ("source_hashes", "input_hashes"):
        if any(plan[field].get(p) != h for p, h in old[field].items()):
            raise RuntimeError("Existing frozen source/input was changed or removed")
    recovery = plan["runtime_recovery"]
    if recovery["previous_plan_hash"] != old["plan_hash"] or recovery["previous_python"] != old["python"]:
        raise RuntimeError("Incorrect recovery provenance")


def load(control):
    plan = load_plan(Path(control) / "plan.json")
    assert_science(plan, read(PREVIOUS / "plan.json"))
    if runtime_snapshot() != plan["runtime_recovery"]["runtime"]:
        raise RuntimeError("Recovery environment fingerprint changed")
    return plan


def prepare(control):
    control = Path(control).absolute()
    if control.exists():
        raise RuntimeError("Never overwrite a recovery campaign")
    snapshot = runtime_snapshot()
    old = load_plan(PREVIOUS / "plan.json")
    if tuple(s["name"] for s in old["systems"]) != ("ministral3_8b", MODEL):
        raise RuntimeError("Unexpected prior model campaign")
    for worker in old["workers"]:
        validate_worker(old, worker["name"])
    with FileLease(PREVIOUS / "supervisor.lock"):
        status = read(PREVIOUS / "status.json")
        if (not status or status["state"] != "needs_review" or status["active"] or status["fatal"] or
                status["blocked_models"] != {MODEL: "preflight failed"}):
            raise RuntimeError("Previous campaign is not the documented stopped dependency failure")
        failure = read(PREVIOUS / "preflight" / MODEL / "result.json")
        check_failure(failure or {}, old["plan_hash"])
        if any(j["model"] == MODEL and (PREVIOUS / "jobs" / j["id"]).exists() for j in old["jobs"]):
            raise RuntimeError("Cannot use a pre-episode recovery after GPT-OSS dispatch")
        failed_run = PREVIOUS / "preflight" / MODEL / "run"
        if any((failed_run / n).exists() for n in ("model.json", "warmup.json", "stress.json", "complete.json")):
            raise RuntimeError("Unexpected model-loaded evidence in the dependency-only failure")
        counts = {"qualification": 0, "test": 0}
        for job in old["jobs"]:
            if job["model"] != "ministral3_8b":
                continue
            result = read(PREVIOUS / "jobs" / job["id"] / "result.json")
            if not result or result["state"] != "complete":
                raise RuntimeError("Completed Ministral evidence is missing")
            previous.four.verify_completion(Path(result["run"]), old, job)
            counts[job["stage"]] += result["episodes"]
        report = read(PREVIOUS / "report/metrics.json")
        if (counts != {"qualification": 144, "test": 1800} or not report or
                report["plan_hash"] != old["plan_hash"] or not report["models"]["ministral3_8b"]["complete"]):
            raise RuntimeError("Ministral final evidence is not complete")
        preserved = dict(read(PREVIOUS / "legacy_inventory.json")["files"])
        for path in PREVIOUS.rglob("*"):
            if path.is_file() and (path.suffix in {".json", ".log", ".md"} or path.name.endswith(".json.gz")):
                preserved[str(path)] = sha(path)
        if any(not Path(p).is_file() or sha(p) != h for p, h in preserved.items()):
            raise RuntimeError("Historical evidence changed")
    control.mkdir(parents=True)
    write_json(control / "legacy_inventory.json", {"files": preserved, "all_verified_at_prepare": True,
        "prior_ministral_audited": counts, "failed_preflight_preserved": True, "time": time.time()})
    plan = recovered_plan(old, snapshot)
    for path in (Path(__file__), ROOT / "repo/tests/test_b_lite_gptoss_env.py"):
        plan["source_hashes"][str(path)] = digest(path.read_text())
    for path in (PREVIOUS / "plan.json", PREVIOUS / "status.json", PREVIOUS / "preflight" / MODEL / "result.json",
                 PREVIOUS / "report/metrics.json", control / "legacy_inventory.json", ROOT / "env_gptoss/pyvenv.cfg"):
        plan["input_hashes"][str(path)] = digest(path.read_bytes().hex())
    assert_science(plan, old)
    plan["plan_hash"] = digest(plan)
    write_json(control / "plan.json", plan)
    write_json(control / "matrix.json", {"plan_hash": plan["plan_hash"], "model": MODEL,
        "formal_total": 1800, "development_total": 144, "workers": plan["workers"],
        "prior_ministral_report": str(PREVIOUS / "report/metrics.json"), "prior_models_rerun": False})
    load(control)
    print(json.dumps({"control": str(control), "plan_hash": plan["plan_hash"], "python": plan["python"],
                      "formal_total": 1800, "development_total": 144}), flush=True)


def validation_commands(plan):
    # Full cross-model regressions need the other models' optional tokenizers.
    # Keep them in their existing environment, and ALSO run all B-lite/GPT-OSS
    # regressions (including its real tokenizer) in the actual target runtime.
    modules = sorted(p.stem for p in (ROOT / "repo/tests").glob("test_gptoss*.py"))
    modules += sorted(p.stem for p in (ROOT / "repo/tests").glob("test_b_lite*.py")
                      if p.stem != "test_b_lite_remaining")
    modules += ["test_b_lite_remaining.RealHarmonyTemplateTests", "test_b_lite_remaining.RemainingQueueTests"]
    return [("full_cpu", [str(ROOT / "env_modern/bin/python"), "-m", "unittest", "discover",
                           "-s", str(ROOT / "repo/tests"), "-v"]),
            ("target_runtime", [plan["python"], "-m", "unittest", "-v", *modules])]


def validate(control):
    control = Path(control)
    plan = load(control)
    if (control / "unit.json").exists():
        raise RuntimeError("Existing validation is immutable")
    records = []
    for name, command in validation_commands(plan):
        path = control / ("unit_" + name + ".log")
        with path.open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "repo"), str(ROOT / "repo/tests"))),
                         HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"))
        records.append({"name": name, "command": command, "returncode": result.returncode,
                        "log_sha256": sha(path)})
    receipt = {"all_pass": all(r["returncode"] == 0 for r in records), "plan_hash": plan["plan_hash"],
               "runtime": runtime_snapshot(), "suites": records, "time": time.time()}
    write_json(control / "unit.json", receipt)
    if not receipt["all_pass"]:
        raise RuntimeError("Full CPU or target-runtime regression gate failed")


def install():
    OLD_INSTALL()
    previous.four.base.MODELS = (MODEL,)
    previous.four.base.validate = validate


def main():
    previous.CONTROL, previous.MODULE, previous.MODELS = CONTROL, MODULE, (MODEL,)
    previous.load, previous.prepare, previous.install = load, prepare, install
    previous.main()


if __name__ == "__main__":
    main()
