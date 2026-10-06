"""Stop only owned V1 supplement controllers/children; preserve all evidence."""
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
project = Path(__file__).resolve().parents[2]
sessions = {"medtoolbench-supplement-a1": "medtoolbench.supplement.campaign",
    "medtoolbench-supplement-followup-v1": "medtoolbench.supplement.pipeline",
    "medtoolbench-supplement-oracle-a1": "medtoolbench.supplement.oracle",
    "medtoolbench-supplement-oracle-c1": "medtoolbench.supplement.action_oracle"}
def command(pid):
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        return path.read_bytes().replace(b"\x00", b" ").decode()
    except FileNotFoundError:
        return ""
parents = {}
for line in subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines():
    pid, parent = map(int, line.split())
    parents[pid] = parent
roots, present = [], []
for session, module in sessions.items():
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode:
        continue
    present.append(session)
    for value in subprocess.check_output(["tmux", "list-panes", "-s", "-t", session, "-F", "#{pane_pid}"], text=True).split():
        pid = int(value)
        cmd = command(pid)
        if module in cmd:
            if "supplement_a_v1" not in cmd and "supplement_c_v1" not in cmd and "supplement_pipeline_v2" not in cmd:
                raise RuntimeError("Unexpected controller command; refuse to stop it")
            roots.append(pid)
owned = set(roots)
while True:
    more = {pid for pid, parent in parents.items() if parent in owned}
    if more <= owned:
        break
    owned |= more
records = {pid: command(pid) for pid in owned}
inference = [pid for pid, cmd in records.items() if "medtoolbench.supplement.runner" in cmd and "--out" in cmd]
for pid in roots:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
for pid in inference:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
deadline = time.monotonic() + 30
while time.monotonic() < deadline and any(command(pid) for pid in inference):
    time.sleep(.5)
if any(command(pid) for pid in inference):
    raise RuntimeError("Owned inference has not exited gracefully; do not start replacements")
# Orphaned oracle HAPI children do not have an ExitStack after parent SIGTERM.
for pid, cmd in records.items():
    if pid not in roots and pid not in inference and command(pid) == cmd and ("java" in cmd or "apptainer" in cmd):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
for session in present:
    subprocess.run(["tmux", "kill-session", "-t", session], check=True)
a.out.mkdir(parents=True, exist_ok=True)
write_json(a.out / (host + ".json"), {"host": host, "time": time.time(), "stopped_controllers": roots,
    "gracefully_stopped_inference": inference, "owned_processes": records, "sessions": present,
    "reason": "V1 supplemental metadata mismatch discovered during qualification; superseded by V2 before test execution",
    "experimental_files_deleted": False, "primary_study_changed": False})
print(json.dumps({"host": host, "controllers_stopped": len(roots), "inference_stopped": len(inference), "evidence_preserved": True}))
