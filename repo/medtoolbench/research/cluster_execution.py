"""Recorded resource-only extension for an unchanged frozen experiment.

The legacy runtime permits GPU indices 0/1 only. This module supplies the same
exclusive lease for an explicitly allocated physical GPU, including 2/3. The
launcher replaces ONLY run.GpuLease, records that substitution, and calls the
original run.main. No model, batching, inference, tool, grading or recovery
function is replaced. Frozen files and their hashes remain unchanged.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from medtoolbench.storage import digest
from .job_state import write_json
from .runtime import core_fingerprint
from .supervisor import load_plan


def short_host():
    return socket.gethostname().split(".")[0]


def extension_hashes():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("cluster_execution.py", "cluster_supervisor.py")}


def deployment(path):
    config = json.loads(Path(path).read_text())
    declared = config.pop("deployment_hash")
    if digest(config) != declared:
        raise RuntimeError("Deployment manifest changed")
    config["deployment_hash"] = declared
    plan = load_plan(Path(config["study"]))
    if plan["plan_hash"] != config["plan_hash"] or core_fingerprint() != plan["core_hashes"]:
        raise RuntimeError("Frozen study/core changed")
    if config["extension_hashes"] != extension_hashes():
        raise RuntimeError("Resource extension changed since deployment")
    analysis = {name: hashlib.sha256(Path(__file__).with_name(name + ".py").read_bytes()).hexdigest()
                for name in plan["analysis_hashes"]}
    if analysis != plan["analysis_hashes"]:
        raise RuntimeError("Frozen analysis changed")
    workers = config["workers"]
    if len({r["name"] for r in workers}) != len(workers):
        raise RuntimeError("Repeated worker name")
    if len({(r["host"], r["gpu"]) for r in workers}) != len(workers):
        raise RuntimeError("Repeated physical GPU assignment")
    if len({r["uuid"] for r in workers}) != len(workers):
        raise RuntimeError("Repeated physical GPU UUID")
    return config, plan


def gpu_inventory():
    response = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
                                        "--format=csv,noheader,nounits"], text=True)
    return [{"gpu": int(r[0]), "uuid": r[1].strip(), "name": r[2].strip(), "memory_mib": int(r[3])}
            for r in csv.reader(io.StringIO(response)) if r]


def validate_worker(config, name):
    matches = [r for r in config["workers"] if r["name"] == name]
    if len(matches) != 1:
        raise RuntimeError("Unknown worker")
    row = matches[0]
    if row["host"] != short_host():
        raise RuntimeError("Worker launched on the wrong node")
    if not any(g["gpu"] == row["gpu"] and g["uuid"] == row["uuid"] for g in gpu_inventory()):
        raise RuntimeError("Physical GPU index/UUID does not match allocation")
    # SSH is adopted by pam_slurm_adopt into the user's allocated extern step.
    groups = Path("/proc/self/cgroup").read_text()
    if f"/job_{row['slurm_job']}/" not in groups:
        raise RuntimeError("Process is outside the declared Slurm allocation")
    return row


class FileLease:
    """Nonblocking filesystem lock; use a shared path for cross-node ownership."""
    def __init__(self, path):
        self.path, self.stream = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            self.stream = None
            raise
        return self

    def fileno(self):
        return self.stream.fileno()

    def __exit__(self, *exc):
        # Close, rather than explicitly unlocking: an inherited descriptor in a
        # live child must retain ownership if its controller unexpectedly dies.
        if self.stream:
            self.stream.close()
            self.stream = None


class AllocatedGpuLease(FileLease):
    def __init__(self, row):
        if short_host() != row["host"] or os.environ.get("CUDA_VISIBLE_DEVICES") != str(row["gpu"]):
            raise RuntimeError("GPU binding changed before inference")
        root = str(Path(__file__).resolve().parents[3])
        identity = digest([root, socket.gethostname()])[:16]
        # Exactly compatible with the frozen 0/1 lease names on the old node.
        super().__init__(Path("/tmp") / f"medtoolbench-{identity}-gpu{row['gpu']}.lock")


def install_resource_adapter(run_module, row):
    run_module.GpuLease = lambda: AllocatedGpuLease(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--probe-hapi", action="store_true")
    args, remaining = parser.parse_known_args()
    config, plan = deployment(args.deployment)
    row = validate_worker(config, args.worker)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(row["gpu"])
    record = {"worker": row, "host": short_host(), "pid": os.getpid(), "time": time.time(),
              "deployment_hash": config["deployment_hash"], "plan_hash": plan["plan_hash"],
              "extension_hashes": extension_hashes(), "core_hashes": core_fingerprint(),
              "substitution": "run.GpuLease only; all scientific functions remain original",
              "cuda_visible_devices": str(row["gpu"])}
    if args.probe:
        if remaining or args.probe.exists():
            raise RuntimeError("Probe requires a new output path and no inference arguments")
        with AllocatedGpuLease(row):
            import torch
            import transformers
            if torch.cuda.device_count() != 1:
                raise RuntimeError("Expected precisely one visible CUDA device")
            props = torch.cuda.get_device_properties(0)
            actual_uuid = str(props.uuid)
            if actual_uuid.removeprefix("GPU-") != row["uuid"].removeprefix("GPU-"):
                raise RuntimeError("CUDA and nvidia-smi GPU identities disagree")
            tensor = torch.arange(8, dtype=torch.float32, device="cuda:0")
            value = (tensor * tensor).sum().item()
            if value != 140.0:
                raise RuntimeError("CUDA arithmetic probe failed")
            record.update(torch=torch.__version__, transformers=transformers.__version__,
                          gpu=props.name, cuda_uuid=actual_uuid, cuda_probe=True,
                          model_inference=False, counts_toward_benchmark=False)
            if args.probe_hapi:
                from .fhir import HapiProcess
                with HapiProcess(plan["image"], args.probe.parent / (args.worker + "-hapi")) as store:
                    record["hapi_probe"] = store.snapshot() == []
                    if not record["hapi_probe"]:
                        raise RuntimeError("Fresh HAPI sandbox not empty")
        args.probe.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.probe, record)
        print(json.dumps(record), flush=True)
        return
    if "--out" not in remaining:
        raise RuntimeError("Missing original run output argument")
    output = Path(remaining[remaining.index("--out") + 1])
    placement = output.parent / (output.name + ".placement.json")
    if output.exists() or placement.exists():
        raise RuntimeError("Never overwrite an attempt or its placement record")
    write_json(placement, record)
    from . import run
    install_resource_adapter(run, row)
    sys.argv = [str(Path(run.__file__)), *remaining]
    run.main()


if __name__ == "__main__":
    main()
