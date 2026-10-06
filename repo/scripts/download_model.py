"""Download public weights/metadata, never call a model inference service."""
import argparse
import fnmatch
import json
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("repo_id")
p.add_argument("destination", type=Path)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
a.destination = a.destination.resolve()
if not a.destination.is_relative_to(root):
    raise SystemExit("Model destination must be inside this project")
ignored = ["*.gguf", "*.onnx", "original/*", "*.pth", "*.pt", "*.bin"]
info = HfApi().model_info(a.repo_id, files_metadata=True)
files = [f for f in info.siblings if not any(fnmatch.fnmatch(f.rfilename, x) for x in ignored)]
if any(f.size is None for f in files):
    raise SystemExit("Cannot admit a download with unknown file sizes")
download_bytes = sum(f.size for f in files)
used = int(subprocess.check_output(["du", "-s", "-B1", str(root)], text=True).split()[0])
if download_bytes > 40_000_000_000 or used + download_bytes > 450_000_000_000:
    raise SystemExit(f"Storage admission failed: project={used}, download={download_bytes}, budget=500000000000")
print(json.dumps({"project_bytes": used, "download_bytes_upper_bound": download_bytes,
                  "budget_bytes": 500_000_000_000}), flush=True)
path = snapshot_download(a.repo_id, revision=info.sha, local_dir=a.destination,
    ignore_patterns=ignored, max_workers=4)
(a.destination / "DOWNLOAD_MANIFEST.json").write_text(json.dumps({"repo_id": a.repo_id,
    "revision": info.sha, "local_path": str(a.destination.resolve()),
    "model_inference_api": False, "download_bytes_upper_bound": download_bytes,
    "project_budget_bytes": 500_000_000_000}, indent=2) + "\n")
print(path)
