"""Atomic job records and terminal dependency checks shared by controllers."""
import json
import os
import time
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def dependency_state(path, campaign=False):
    path = Path(path)
    for name in ("failed.json", "infrastructure_failure.txt", "cancelled.json"):
        if (path / name).exists():
            return "failed", f"Dependency has terminal marker {name}"
    progress = path / "progress.json"
    if campaign and progress.exists():
        try:
            rows = json.loads(progress.read_text())
        except json.JSONDecodeError:
            return "waiting", "Dependency is updating a legacy non-atomic progress record"
        if isinstance(rows, list) and any(r.get("returncode", 0) != 0 or
                r.get("audit") is not None and not r["audit"].get("all_pass") for r in rows):
            return "failed", "Dependency campaign contains a failed model or audit"
    if (path / "complete.json").exists():
        return "complete", "Dependency completed"
    return "waiting", "Dependency is not complete"


def wait_dependency(path, output, campaign=False, timeout=86400, poll=15):
    started = time.monotonic()
    while True:
        state, reason = dependency_state(path, campaign)
        write_json(Path(output) / "status.json", {"state": state, "reason": reason, "dependency": str(path), "heartbeat": time.time()})
        if state == "failed":
            raise RuntimeError(reason)
        if state == "complete":
            return
        if time.monotonic() - started >= timeout:
            raise TimeoutError("Dependency did not complete within the declared time limit")
        time.sleep(poll)
