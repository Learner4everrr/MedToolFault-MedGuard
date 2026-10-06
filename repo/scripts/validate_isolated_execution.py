"""Recorded full unit suite and real-HAPI malformed-input regression."""
import argparse
import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.storage import digest
from medtoolbench.research.contracts import service
from medtoolbench.research.fhir import HapiProcess, normalized_state
from medtoolbench.research.job_state import write_json
from medtoolbench.research.run import Episode
from medtoolbench.supplement.input_guard import parse_action
from medtoolbench.supplement.isolated_manifest import load
from medtoolbench.supplement.runner import load_plan
from medtoolbench.supplement.transport import CONDITIONS, CallBudget, FactorClient, StressProxy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    manifest = load(a.manifest)
    root = a.manifest.parent / "validation"
    if root.exists():
        raise RuntimeError("Never overwrite validation evidence")
    root.mkdir()
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover("repo/tests")
    with (root / "unit.log").open("x") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    proof = {"all_pass": result.wasSuccessful(), "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "seconds": time.monotonic() - started, "manifest_hash": manifest["manifest_hash"], "time": time.time()}
    write_json(root / "unit.json", proof)
    print(json.dumps(proof), flush=True)
    if not proof["all_pass"]:
        raise SystemExit(1)
    plan = load_plan(Path(manifest["campaigns"][0]))
    checks = []
    try:
        with HapiProcess(plan["image"], root / "hapi") as store:
            fixture = [{"resourceType": "Patient", "id": "input-guard-test"}]
            body = service("input-guard-test", "4548-4")
            task = {"patient": "input-guard-test", "hazards": ["clean"],
                    "contract": {"required_creations": [body]}}
            for condition in CONDITIONS:
                before = store.reload(fixture)
                proxy = StressProxy(store, task, "clean")
                budget = CallBudget(proxy.call, 48)
                episode = Episode.__new__(Episode)
                episode.client = FactorClient(budget, condition)
                episode.messages, episode.generations, episode.parsed_actions = [], [], []
                bad = {"method": "POST", "path": "/ServiceRequest", "body": {**body, "meta": None}}
                good = {"method": "POST", "path": "/ServiceRequest", "body": body}
                with patch("medtoolbench.research.run.parse_action", parse_action):
                    episode.advance({"text": json.dumps(bad)})
                    rejected = json.loads(episode.messages[-1]["content"])["status"] == 400
                    unchanged = normalized_state(before) == normalized_state(store.snapshot())
                    no_raw = budget.used == 0 and not proxy.trace
                    episode.advance({"text": json.dumps(good)})
                accepted = json.loads(episode.messages[-1]["content"])["status"] < 400
                writes = [r for r in store.snapshot() if r["resourceType"] == "ServiceRequest"]
                check = {"condition": condition, "invalid_rejected": rejected, "invalid_not_dispatched": no_raw,
                    "state_unchanged_after_invalid": unchanged, "subsequent_valid_accepted": accepted,
                    "exactly_one_valid_write": len(writes) == 1,
                    "all_pass": rejected and unchanged and no_raw and accepted and len(writes) == 1}
                checks.append(check)
                if not check["all_pass"]:
                    raise RuntimeError("Real-HAPI parser regression failed")
        write_json(root / "hapi.json", {"all_pass": True, "checks": checks,
            "manifest_hash": manifest["manifest_hash"], "time": time.time()})
    except BaseException:
        import traceback
        write_json(root / "hapi.json", {"all_pass": False, "checks": checks, "error": traceback.format_exc(),
            "manifest_hash": manifest["manifest_hash"], "time": time.time()})
        raise
    print(json.dumps({"real_hapi_checks": len(checks), "all_pass": True}), flush=True)


if __name__ == "__main__":
    main()
