"""Launch only this node's declared workers in a project-specific tmux session."""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from medtoolbench.research.cluster_execution import deployment, short_host, validate_worker
from medtoolbench.research.job_state import write_json

p = argparse.ArgumentParser()
p.add_argument("--deployment", type=Path, required=True)
a = p.parse_args()
path = a.deployment.resolve()
config, plan = deployment(path)
handoff = json.loads((path.parent / "handoff.json").read_text())
if not handoff.get("legacy_controllers_stopped") or handoff["plan_hash"] != plan["plan_hash"]:
    raise RuntimeError("Handoff is not complete")
host = short_host()
rows = [r for r in config["workers"] if r["host"] == host]
if not rows:
    raise RuntimeError("No workers allocated on this host")
for row in rows:
    validate_worker(config, row["name"])
session = "medtoolbench-sixgpu"
if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
    raise RuntimeError("Project tmux session already exists; inspect rather than duplicating it")
root = Path(config["study"]).parents[1]
subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "control", "-c", str(root)], check=True)
subprocess.run(["tmux", "set-option", "-t", session, "remain-on-exit", "on"], check=True)
commands = []
for row in rows:
    command = ["env", "PYTHONPATH=" + str(root / "repo"), config["inference_python"], "-u", "-m",
               "medtoolbench.research.cluster_supervisor", "--deployment", str(path), "--worker", row["name"]]
    subprocess.run(["tmux", "new-window", "-t", session, "-n", f"gpu{row['gpu']}", "-c", str(root),
                    "exec " + shlex.join(command)], check=True)
    commands.append({"worker": row["name"], "command": command})
record = {"host": host, "tmux_session": session, "workers_launched": len(rows),
          "commands": commands, "time": time.time(), "cwd": str(root)}
write_json(path.parent / (host + "-launch.json"), record)
print(json.dumps(record), flush=True)
