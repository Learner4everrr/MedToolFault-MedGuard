"""Close preparation gates, freeze once, then actually execute both primary workers."""
import argparse
import json
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .job_state import dependency_state, write_json
from .runtime import core_fingerprint
from .supervisor import stop_owned
from .validation_audit import audit as audit_validation


def main():
    p = argparse.ArgumentParser()
    for name in ("benchmark", "validation", "registry", "qualifications", "image", "study", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Preparation controller received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    if a.out.exists() or a.study.exists():
        raise SystemExit("Preparation and study output directories must be new")
    a.out.mkdir(parents=True)
    expected = core_fingerprint()
    qualifications = json.loads(a.qualifications.read_text())
    dependencies = [a.validation, *map(Path, qualifications.values())]
    last_change = {str(path): time.monotonic() for path in dependencies}
    signatures = {}
    children, streams = [], []
    started = time.monotonic()
    try:
        while True:
            if expected != core_fingerprint():
                raise RuntimeError("Core changed during preparation; inspect and create a new preparation version")
            waiting = []
            for path in dependencies:
                state, reason = dependency_state(path)
                if state == "failed":
                    raise RuntimeError(str(path) + ": " + reason)
                if state == "complete":
                    continue
                progress = path / "progress.json"
                sig = (progress.stat().st_mtime_ns, progress.stat().st_size) if progress.exists() else None
                if sig != signatures.get(str(path)):
                    last_change[str(path)] = time.monotonic()
                    signatures[str(path)] = sig
                if time.monotonic() - last_change[str(path)] > 7200:
                    raise TimeoutError("Preparation dependency made no progress for 2 hours: " + str(path))
                waiting.append(str(path))
            write_json(a.out / "status.json", {"state": "waiting" if waiting else "auditing", "waiting": waiting, "heartbeat": time.time()})
            if not waiting:
                break
            if time.monotonic() - started > 28800:
                raise TimeoutError("Preparation exceeded its declared 8-hour deadline")
            time.sleep(15)
        proof = audit_validation(a.validation, a.benchmark)
        if not proof["all_pass"]:
            raise RuntimeError("Full oracle replay/completeness audit failed")
        from .freeze_study import freeze
        plan = freeze(a.benchmark, a.validation, a.registry, qualifications, a.image, a.study)
        write_json(a.out / "status.json", {"state": "primary_running", "study": str(a.study.resolve()),
                                           "plan_hash": plan["plan_hash"], "heartbeat": time.time()})
        for worker in (0, 1):
            stream = (a.out / f"worker{worker}.log").open("w")
            streams.append(stream)
            children.append(subprocess.Popen([sys.executable, "-u", "-m", "medtoolbench.research.supervisor",
                "--study", str(a.study.resolve()), "--worker", str(worker)], stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True))
        while any(c.poll() is None for c in children):
            write_json(a.out / "status.json", {"state": "primary_running", "worker_returncodes": [c.poll() for c in children],
                                               "study": str(a.study.resolve()), "heartbeat": time.time()})
            time.sleep(15)
        if any(c.returncode != 0 for c in children) or not (a.study / "complete.json").exists():
            raise RuntimeError("Primary execution finished with missing/failed shards; see worker/job records")
        write_json(a.out / "complete.json", {"state": "primary_complete", "study": str(a.study.resolve()),
                                             "whole_research_project_complete": False, "time": time.time()})
    except BaseException:
        write_json(a.out / "failed.json", {"state": "failed", "error": traceback.format_exc(), "time": time.time()})
        raise
    finally:
        for child in children:
            stop_owned(child)
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
