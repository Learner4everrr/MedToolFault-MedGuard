"""Serial local-model development queue on one assigned GPU; audit every run."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from medtoolbench.storage import digest
from .audit import audit
from .job_state import wait_dependency, write_json

CORE = ("contracts.py", "fhir.py", "models.py", "protocol.py", "transport.py", "run.py", "oracle.py", "input_validation.py", "runtime.py", "build.py", "job_state.py")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--gpu", type=int, choices=[0, 1], required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--after-campaign", action="store_true", help="The dependency is a completed campaign, not a single run")
    p.add_argument("--conditions", nargs="+", choices=["vanilla", "warning", "retry_only", "medguard", "oracle_hint"], default=["vanilla", "medguard"])
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Campaign directory exists")
    a.out.mkdir(parents=True)
    systems = {r["name"]: r for r in json.loads(a.registry.read_text())["systems"]}
    if len(set(a.names)) != len(a.names) or not set(a.names) <= systems.keys():
        raise SystemExit("Unknown or repeated model names")
    package = Path(__file__).parent
    fingerprint = {name: digest((package / name).read_text()) for name in CORE}
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(phase="development only", source_hashes=fingerprint, inference_api=False)
    (a.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"state": "waiting_for_prior_run", "path": str(a.after)}), flush=True)
    try:
        wait_dependency(a.after, a.out, campaign=a.after_campaign)
    except (RuntimeError, TimeoutError) as error:
        write_json(a.out / "failed.json", {"state": "failed", "reason": str(error), "time": time.time()})
        raise SystemExit(str(error))
    if a.after_campaign:
        prior = json.loads((a.after / "complete.json").read_text())
        if not prior.get("models"):
            raise SystemExit("Prior campaign has no model results")
        for row in prior["models"]:
            if row["returncode"] != 0 or not row.get("audit", {}).get("all_pass"):
                raise SystemExit("Prior campaign did not pass its recorded checks")
            if not audit(a.after / row["model"], a.benchmark)["all_pass"]:
                raise SystemExit("Prior campaign failed refreshed replay audit")
    elif not audit(a.after, a.benchmark)["all_pass"]:
        raise SystemExit("Prior run did not pass replay/completeness audit")
    results = []
    for name in a.names:
        if fingerprint != {n: digest((package / n).read_text()) for n in CORE}:
            raise SystemExit("Core implementation changed while the campaign was queued; create a new version")
        r = systems[name]
        out = a.out / name
        command = [sys.executable, "-u", "-m", "medtoolbench.research.run", "--benchmark", str(a.benchmark.resolve()),
                   "--validation", str(a.validation.resolve()), "--image", str(a.image.resolve()),
                   "--model", r["path"], "--out", str(out.resolve()), "--split", "dev", "--per-template", "1",
                   "--batch-size", str(r["batch_size"]), "--max-new-tokens", "2048",
                   "--conditions", *a.conditions]
        if r["thinking"]:
            command.append("--thinking")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
        print(json.dumps({"state": "running", "model": name, "command": command}), flush=True)
        with (a.out / (name + ".log")).open("w") as log:
            proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        result = {"model": name, "returncode": proc.returncode,
                  "audit": audit(out, a.benchmark) if (out / "complete.json").exists() else None}
        results.append(result)
        write_json(a.out / "progress.json", results)
        # Model accuracy is not a stopping criterion. Infrastructure/audit failure is.
        if proc.returncode != 0 or not result["audit"] or not result["audit"]["all_pass"]:
            write_json(a.out / "failed.json", {"state": "failed", "model": name, "returncode": proc.returncode,
                                               "reason": "Infrastructure or replay audit failure", "time": time.time()})
            raise SystemExit(f"Infrastructure or audit failure for {name}; see its retained log")
    (a.out / "complete.json").write_text(json.dumps({"models": results, "phase": "development only"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
