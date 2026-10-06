"""Read-only evidence/lease checks, then an explicit administrative-stop receipt.

Does not start or resume experiments, change old results, or conceal the missed
development-only stop. Partial formally assigned rows remain quarantined.
"""
import json
import os
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

from medtoolbench.supplement.b_lite import ROOT, read, sha
from medtoolbench.supplement.b_lite_four import load, verify_completion
from medtoolbench.research.cluster_execution import FileLease, AllocatedGpuLease, validate_worker
from medtoolbench.research.job_state import write_json

CONTROL = ROOT / "data/b_lite_llama_four_gpu_v2"
PAUSE = ROOT / "data/b_lite_pause_after_llama_dev_v1"


def main():
    plan = load(CONTROL)
    if PAUSE.exists():
        raise RuntimeError("Handoff receipt is immutable")
    old_status = read(CONTROL / "status.json")
    if old_status["active"] or Path("/proc") .joinpath(str(read(CONTROL / "started.json")["pid"])).exists():
        raise RuntimeError("Original controller has not exited")
    development, completed_formal, partial_formal = 0, 0, 0
    interrupted, files = [], {}
    with ExitStack() as stack:
        stack.enter_context(FileLease(CONTROL / "supervisor.lock"))
        for job in plan["jobs"]:
            folder = CONTROL / "jobs" / job["id"]
            if not folder.exists():
                if job["stage"] == "qualification":
                    raise RuntimeError("Missing development shard")
                continue
            stack.enter_context(FileLease(CONTROL / "locks" / (job["id"] + ".lock")))
            result = read(folder / "result.json")
            if not result:
                raise RuntimeError("Unresolved started shard")
            output = Path(result["run"])
            if result["state"] == "complete":
                verify_completion(output, plan, job)
                if job["stage"] == "qualification":
                    development += len(job["episodes"])
                else:
                    completed_formal += len(job["episodes"])
            else:
                if job["stage"] != "test" or result["execution"]["timeout"] != "controller_stop":
                    raise RuntimeError("Unexpected failed shard; separate review required")
                count = len(list((output / "episodes").glob("*.json.gz")))
                partial_formal += count
                interrupted.append({"job": job["id"], "assigned": len(job["episodes"]),
                    "saved_partial": count, "cause": "administrative user-requested stop",
                    "model_failure": False, "partial_rows_pooled": False})
            for path in folder.rglob("*.json*"):
                if path.is_file():
                    files[str(path)] = sha(path)
        for worker in plan["workers"]:
            validate_worker(plan, worker["name"])
            os.environ["CUDA_VISIBLE_DEVICES"] = str(worker["gpu"])
            stack.enter_context(AllocatedGpuLease(worker))
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        gate = read(CONTROL / "gates/llama31_8b.json")
        if development != 144 or not gate["all_pass"] or gate["plan_hash"] != plan["plan_hash"]:
            raise RuntimeError("Development audit/capability gate is not complete")
        for path in (CONTROL / "plan.json", CONTROL / "unit.json", CONTROL / "status.json",
                     CONTROL / "gates/llama31_8b.json", Path(__file__)):
            files[str(path)] = sha(path)
        gpu_snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                                "--format=csv,noheader,nounits"], text=True)
        if any(int(line.split(",")[1].strip()) != 0 for line in gpu_snapshot.splitlines() if line.strip()):
            raise RuntimeError("Allocated GPU memory is not fully released")
        PAUSE.mkdir()
        write_json(PAUSE / "evidence_hashes.json", {"sha256": files, "time": time.time()})
        record = {"state": "paused_ready_for_eight_gpu_handoff", "time": time.time(),
            "instruction": "Finish Llama development only, then stop before switching to eight GPUs",
            "plan_hash": plan["plan_hash"], "development_audited": development,
            "development_gate_passed": True, "controller_exited": True,
            "job_and_gpu_leases_released": True, "gpu_snapshot": gpu_snapshot,
            "automatic_resumption": False, "future_gpu_count": 8,
            "new_resource_manifest_required": True, "entire_experiment_complete": False,
            "stop_before_formal_succeeded": False, "unexpected_formal_started": True,
            "formal_audited_preserved": completed_formal, "formal_partial_quarantined": partial_formal,
            "interrupted_shards": interrupted, "partial_rows_pooled": False,
            "incident": "The initially observed dispatcher SIGSTOP did not remain effective; formal dispatch was later observed. The cause of resumption is unverified. SIGTERM then stopped the controller and all owned workers. No result was deleted or changed.",
            "unused_pause_observer": "b_lite_dev_pause was drafted/tested but never armed or launched",
            "evidence_inventory": str(PAUSE / "evidence_hashes.json"),
            "next_action": "Wait for explicit user handoff. Preserve completed evidence; separately reconcile administratively interrupted shards, without outcome-based selection or blind resume."}
        write_json(PAUSE / "ready_for_eight_gpu_handoff.json", record)
        write_json(PAUSE / "status.json", record)
        print(json.dumps({k: record[k] for k in ("state", "development_audited", "formal_audited_preserved",
            "formal_partial_quarantined", "stop_before_formal_succeeded", "automatic_resumption")}), flush=True)


if __name__ == "__main__":
    main()
