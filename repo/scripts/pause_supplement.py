"""User-requested pause of exactly the current supplement's owned process trees."""
import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from medtoolbench.research.job_state import write_json

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
host = socket.gethostname().split(".")[0]
if host not in {"aga38", "agb03"}:
    raise RuntimeError("Unexpected host")
sessions = {
    "medtoolbench-supplement-a2": ("medtoolbench.supplement.campaign", "supplement_a_v2/plan.json"),
    "medtoolbench-supplement-followup-v1": ("medtoolbench.supplement.pipeline", "supplement_pipeline_v3/pipeline.json"),
    "medtoolbench-supplement-oracle-a2": ("medtoolbench.supplement.corrected_oracle", "supplement_a_v2/plan.json"),
    "medtoolbench-supplement-oracle-c2": ("medtoolbench.supplement.corrected_oracle", "supplement_c_v2/plan.json"),
}

def identity(pid):
    try:
        root = Path("/proc") / str(pid)
        cmd = (root / "cmdline").read_bytes().replace(b"\x00", b" ").decode()
        start = (root / "stat").read_text().rsplit(")", 1)[1].split()[19]
        return {"command": cmd, "start": start} if cmd else None
    except (FileNotFoundError, ProcessLookupError):
        return None

parents = {}
for line in subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines():
    pid, parent = map(int, line.split())
    parents[pid] = parent
roots, present = [], []
for session, (module, plan) in sessions.items():
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode:
        continue
    present.append(session)
    panes = subprocess.check_output(["tmux", "list-panes", "-s", "-t", session, "-F", "#{pane_pid} #{pane_dead}"], text=True)
    for line in panes.splitlines():
        value, dead = line.split()
        if dead == "1":
            continue
        pid = int(value)
        item = identity(pid)
        if not item:
            raise RuntimeError("Cannot inspect live pane PID; run outside the isolated PID namespace")
        if module in item["command"]:
            if plan not in item["command"]:
                raise RuntimeError("Unexpected plan in owned session")
            roots.append(pid)
        elif "bash" not in item["command"]:
            raise RuntimeError("Unexpected process in owned session")
owned = set(roots)
while True:
    more = {pid for pid, parent in parents.items() if parent in owned}
    if more <= owned:
        break
    owned |= more
records = {pid: item for pid in owned if (item := identity(pid))}
inference = [pid for pid, item in records.items() if "medtoolbench.supplement.corrected_runner" in item["command"] and "--out" in item["command"]]
a.out.mkdir(parents=True, exist_ok=True)
receipt = a.out / (host + ".json")
if receipt.exists():
    raise RuntimeError("Do not overwrite an earlier pause receipt")
record = {"host": host, "requested_at": time.time(), "reason": "User requested pause before extending allocation time",
    "sessions": present, "controllers": roots, "inference": inference, "owned_processes": records,
    "state": "pausing", "experimental_evidence_deleted": False, "primary_study_changed": False}
write_json(receipt, record)

def alive(pid):
    now = identity(pid)
    return now is not None and now["start"] == records[pid]["start"]

def send(pid, sig):
    if alive(pid):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass

# Stop all schedulers first so no retry or follow-up job can be dispatched.
for pid in roots:
    send(pid, signal.SIGTERM)
for pid in inference:
    send(pid, signal.SIGTERM)
deadline = time.monotonic() + 90
while time.monotonic() < deadline and any(alive(pid) for pid in inference):
    time.sleep(.5)
record["forced_inference_termination"] = [pid for pid in inference if alive(pid)]
for pid in record["forced_inference_termination"]:
    send(pid, signal.SIGKILL)
# Reap only descendants captured from this project's verified controller trees.
for pid in records:
    send(pid, signal.SIGTERM)
deadline = time.monotonic() + 10
while time.monotonic() < deadline and any(alive(pid) for pid in records):
    time.sleep(.5)
record["forced_descendant_termination"] = [pid for pid in records if alive(pid)]
for pid in record["forced_descendant_termination"]:
    send(pid, signal.SIGKILL)
for session in present:
    # A terminated final pane can remove its session before this check.
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
record.update(state="paused", stopped_at=time.time())
write_json(receipt, record)
for folder in (Path("data/supplement_a_v2/workers"), Path("data/supplement_pipeline_v3")):
    for file in folder.glob(host + "-gpu*.json"):
        previous = json.loads(file.read_text())
        write_json(a.out / (folder.parent.name + "--" + file.name), previous)
        write_json(file, {**previous, "state": "user_paused", "heartbeat": time.time(), "pause_receipt": str(receipt.resolve())})
print(json.dumps({"host": host, "state": "paused", "controllers": len(roots), "inference": len(inference),
    "forced_inference_termination": record["forced_inference_termination"], "evidence_preserved": True}), flush=True)
