"""Explicit, logged shutdown of this study's legacy controller, preserving runs."""
import argparse
import json
import os
import signal
import time
from pathlib import Path

from medtoolbench.research.audit import audit
from medtoolbench.research.cluster_execution import AllocatedGpuLease, deployment, short_host
from medtoolbench.research.job_state import write_json
from medtoolbench.research.status import status


def proc_record(pid):
    root = Path("/proc") / str(pid)
    try:
        stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {"pid": pid, "state": stat[0], "start_ticks": stat[19],
                "uid": root.stat().st_uid, "cwd": str((root / "cwd").resolve()),
                "argv": (root / "cmdline").read_bytes().decode().strip("\0").split("\0")}
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deployment", type=Path, required=True)
    p.add_argument("--legacy-controller-pid", type=int, required=True)
    a = p.parse_args()
    config, plan = deployment(a.deployment)
    control, root = a.deployment.resolve().parent, Path(config["study"])
    if (control / "handoff.json").exists() or (control / "handoff-request.json").exists():
        raise RuntimeError("A handoff was already requested; inspect it before taking another action")
    if short_host() != "aga38":
        raise RuntimeError("Legacy handoff must run on aga38")
    for worker in (r for r in config["workers"] if r["host"] == "agb03"):
        proof = json.loads((control / "probes" / (worker["name"] + ".json")).read_text())
        if not proof.get("cuda_probe") or proof["worker"] != worker:
            raise RuntimeError("Remote GPU probe missing or mismatched")
    if not json.loads((control / "probes/agb03-gpu2.json").read_text()).get("hapi_probe"):
        raise RuntimeError("Remote HAPI startup has not passed")
    for label in ("busy", "free"):
        if not json.loads((control / f"cross-node-lock-{label}.json").read_text())["all_pass"]:
            raise RuntimeError("Cross-node lock/release test has not passed")
    legacy = proc_record(a.legacy_controller_pid)
    if (not legacy or legacy["uid"] != os.getuid()
            or "medtoolbench.research.prepare_and_execute" not in legacy["argv"]
            or Path(legacy["cwd"]) != root.parents[1] / "repo"):
        raise RuntimeError("PID is not this user's expected project preparation controller")
    argv = legacy["argv"]
    if (Path(legacy["cwd"]) / argv[argv.index("--study") + 1]).resolve() != root:
        raise RuntimeError("Controller belongs to a different study")
    children_path = Path(f"/proc/{a.legacy_controller_pid}/task/{a.legacy_controller_pid}/children")
    children = [proc_record(int(pid)) for pid in children_path.read_text().split()]
    children = [r for r in children if r]
    if len(children) != 2 or any("medtoolbench.research.supervisor" not in r["argv"] for r in children):
        raise RuntimeError("Unexpected legacy child process topology")
    before = status(root.parent)
    request = {"plan_hash": plan["plan_hash"], "host": short_host(), "controller": legacy,
               "children": children, "before": before["primary"], "requested_at": time.time(),
               "reason": "User-authorized addition of four GPUs; unchanged frozen study"}
    write_json(control / "handoff-request.json", request)
    print(json.dumps({"state": "stopping_legacy_controller", "pid": a.legacy_controller_pid}), flush=True)
    os.kill(a.legacy_controller_pid, signal.SIGTERM)
    deadline = time.monotonic() + 180
    tracked = [legacy, *children]
    while any((current := proc_record(r["pid"])) and current["start_ticks"] == r["start_ticks"] for r in tracked):
        if time.monotonic() > deadline:
            raise RuntimeError("Legacy controllers did not stop; no handoff marker issued")
        time.sleep(2)
    for worker in (r for r in config["workers"] if r["host"] == short_host()):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker["gpu"])
        with AllocatedGpuLease(worker):
            pass
    proofs = {}
    for job in plan["jobs"]:
        folder = root / "jobs" / job["id"]
        if (folder / "result.json").exists():
            continue
        attempts = sorted(p for p in folder.glob("attempt*") if p.is_dir() and p.name[7:].isdigit())
        if not attempts:
            continue
        source = attempts[-1]
        proof = audit(source, Path(plan["benchmark"]))
        if proof["errors"] or proof["unexpected"]:
            raise RuntimeError("Partial evidence failed the independent handoff audit")
        if not (source / "complete.json").exists() and not (source / "infrastructure_failure.txt").exists():
            raise RuntimeError("Stopped attempt lacks a clean termination record")
        proofs[job["id"]] = {"source": str(source), "episodes": proof["episodes"],
                              "expected": proof["expected"], "errors": 0, "unexpected": 0}
    result = {"legacy_controllers_stopped": True, "plan_hash": plan["plan_hash"],
              "deployment_hash": config["deployment_hash"], "finished_at": time.time(),
              "partial_evidence_audits": proofs, "post_stop_status": status(root.parent)["primary"],
              "preservation": "Complete audited batches reused; incomplete batches rerun; no artifacts deleted"}
    write_json(control / "handoff.json", result)
    write_json(root.parent / "primary_preparation_v2/superseded_by.json",
               {"deployment": str(a.deployment.resolve()), "handoff": str(control / "handoff.json"),
                "reason": "Intentional resource-controller replacement, not a scientific failure"})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
