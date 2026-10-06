"""Read/write a dedicated test lock, never a production job lock."""
import argparse
import json
import socket
from pathlib import Path
from medtoolbench.research.cluster_execution import FileLease
from medtoolbench.research.job_state import write_json

p = argparse.ArgumentParser()
p.add_argument("--path", type=Path, required=True)
p.add_argument("--expect", choices=["busy", "free"], required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
try:
    with FileLease(a.path):
        observed = "free"
except BlockingIOError:
    observed = "busy"
report = {"host": socket.gethostname(), "expected": a.expect, "observed": observed,
          "all_pass": observed == a.expect}
write_json(a.out, report)
print(json.dumps(report), flush=True)
raise SystemExit(0 if report["all_pass"] else 2)
