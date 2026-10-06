"""Record exact live ownership before terminating only the old dispatcher."""
import json
import time
from pathlib import Path
from medtoolbench.supplement.storage_recovery import atomic_json

root = Path(__file__).resolve().parents[2]
control = root / "data/gptoss_replacement_v1"
out = control / "handoff.json"
if out.exists():
    raise RuntimeError("Handoff snapshot already exists; do not overwrite")
old = root / "data/supplement_storage_recovery_v1"
status = json.loads((old / "status.json").read_text())
pid = status["pid"]
cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
if "medtoolbench.supplement.storage_recovery run" not in cmdline:
    raise RuntimeError("Controller PID identity changed")
workers = []
for file in sorted((old / "workers").glob("*.json")):
    row = json.loads(file.read_text())
    if row.get("state") != "running":
        continue
    proc = Path(f"/proc/{row['pid']}/cmdline")
    if not proc.exists():
        continue
    row["observed_cmdline"] = proc.read_bytes().replace(b"\0", b" ").decode()
    row["config"] = json.loads((Path(row["out"]) / "config.json").read_text())
    if row["config"]["job"]["model"] == "gemma3_12b":
        raise RuntimeError("Live Gemma found; stop the exact runner first")
    if row["out"] not in row["observed_cmdline"]:
        raise RuntimeError("Child PID identity changed")
    workers.append(row)
atomic_json(out, {"time": time.time(), "reason": "User explicitly retires Gemma and replaces all its future roles with GPT-OSS-20B",
    "old_controller_pid": pid, "old_controller_cmdline": cmdline, "old_status": status,
    "live_children_preserved": workers, "old_control": str(old), "inference_api": False})
print(json.dumps({"snapshot": str(out), "controller_pid": pid,
    "live_non_gemma_children": [(r["pid"], r["worker"]["name"]) for r in workers]}), flush=True)
