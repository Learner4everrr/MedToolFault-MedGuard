"""B-lite development gates, isolated scheduling, auditing and analysis.

No mutation of old campaigns. Completed shards are checkpoints; a failed shard
is retained (including partial rows), quarantined, and never accuracy-retried.
"""
import argparse
import collections
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.build import load_gzip
from medtoolbench.research.cluster_execution import FileLease, AllocatedGpuLease, validate_worker
from medtoolbench.research.job_state import write_json
from medtoolbench.research.supervisor import stop_owned
from .b_lite import ROOT, CONTROL, MODELS, HAZARDS, POLICIES, SETTINGS, read, sha, patient_effect
from .runner import SampleModel, load_plan
from .memory_execution import verify_completion


class IntegrityError(RuntimeError):
    pass


def load(control):
    plan = load_plan(Path(control) / "plan.json")
    if tuple(s["name"] for s in plan["systems"]) != MODELS:
        raise IntegrityError("Undeclared model substitution")
    if any(s["batch_size"] != 2 for s in plan["systems"]):
        raise IntegrityError("Undeclared batch change")
    if any(j["settings"] != SETTINGS for j in plan["jobs"]):
        raise IntegrityError("Scientific budgets changed")
    return plan


def require_unit(control, plan):
    unit = read(Path(control) / "unit.json")
    if not unit or not unit.get("all_pass") or unit.get("plan_hash") != plan["plan_hash"]:
        raise IntegrityError("CPU regression gate not satisfied")


def validate(control):
    plan = load(control)
    path = Path(control) / "unit.json"
    if path.exists():
        raise IntegrityError("Existing validation is immutable")
    command = [plan["python"], "-m", "unittest", "discover", "-s", str(ROOT / "repo/tests"), "-v"]
    with (Path(control) / "unit.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                env=dict(os.environ, PYTHONPATH=str(ROOT / "repo")))
    write_json(path, {"all_pass": result.returncode == 0, "returncode": result.returncode,
                     "plan_hash": plan["plan_hash"], "command": command,
                     "log_sha256": sha(Path(control) / "unit.log"), "time": time.time()})
    if result.returncode:
        raise RuntimeError("CPU regressions failed")


def preflight(control, model_name, worker_name, output):
    plan = load(control)
    require_unit(control, plan)
    worker = validate_worker(plan, worker_name)
    system = next(s for s in plan["systems"] if s["name"] == model_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker["gpu"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        with AllocatedGpuLease(worker):
            model = SampleModel(Path(system["path"]), False, SETTINGS["seed"])
            if model.kind != system["model_type"]:
                raise IntegrityError("Local model architecture mismatch")
            t, tok = model.torch, model.tokenizer
            write_json(output / "model.json", model.metadata)
            prompt = model.prompt([{"role": "user", "content": 'Return only {"ready":true}.'}])
            warm = model.generate([prompt, prompt], 32)
            write_json(output / "progress.json", {"stage": "memory_stress", "heartbeat": time.time()})
            token = tok.encode(" record", add_special_tokens=False)[0]
            ids = t.full((2, 14336), token, dtype=t.long, device="cuda:0")
            mask = t.ones_like(ids)
            t.cuda.synchronize()
            t.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with t.inference_mode():
                generated = model.model.generate(input_ids=ids, attention_mask=mask,
                    max_new_tokens=2048, min_new_tokens=2048, do_sample=False,
                    eos_token_id=None, pad_token_id=tok.pad_token_id, use_cache=True)
            t.cuda.synchronize()
            seconds = time.perf_counter() - started
            reserved = t.cuda.max_memory_reserved()
            total = t.cuda.get_device_properties(0).total_memory
            passed = generated.shape == (2, 16384) and reserved / total <= .95
            receipt = {"all_pass": passed, "plan_hash": plan["plan_hash"], "model": model_name,
                "worker": worker, "fixed_batch_size": 2, "prompt_tokens": 14336,
                "forced_output_tokens": 2048, "generated_shape": list(generated.shape),
                "seconds": seconds, "peak_reserved_bytes": reserved,
                "peak_allocated_bytes": t.cuda.max_memory_allocated(), "total_memory_bytes": total,
                "warmup": warm, "synthetic_only": True, "formal_evidence": False,
                "time": time.time()}
            write_json(output / "stress.json", receipt)
            if not passed:
                raise RuntimeError("Full-context stress shape/headroom gate failed")
            write_json(output / "complete.json", receipt)
    except BaseException:
        write_json(output / "failure.json", {"error": traceback.format_exc(), "time": time.time()})
        raise


def run_shard(control, job_id, worker_name, output):
    # Use the ORIGINAL external Episode, tools, parser, budgets, HAPI and grader.
    plan = load(control)
    require_unit(control, plan)
    job = next(j for j in plan["jobs"] if j["id"] == job_id)
    if job["stage"] == "test" and model_gate(control, plan, job["model"]) != "passed":
        raise IntegrityError("Held-out job cannot bypass development gates")
    from . import external_runner

    class RecordedModel(SampleModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.metadata["b_lite"] = {"plan_hash": plan["plan_hash"], "fixed_batch_size": 2,
                "budgets_parser_tools_labels_unchanged": True, "old_rows_pooled": False,
                "fast_kernels_enabled": False, "bitwise_equivalence_claimed": False}

    external_runner.SampleModel = RecordedModel
    sys.argv = [external_runner.__file__, "--plan", str(Path(control) / "plan.json"),
                "--job", job_id, "--worker", worker_name, "--out", str(output)]
    external_runner.main()


def model_gate(control, plan, model):
    control = Path(control)
    stress = read(control / "preflight" / model / "complete.json")
    if not stress:
        return "failed" if (control / "preflight" / model / "failure.json").exists() else "waiting"
    if not stress.get("all_pass") or stress.get("plan_hash") != plan["plan_hash"]:
        raise IntegrityError("Invalid stress receipt")
    qualified = [j for j in plan["jobs"] if j["stage"] == "qualification" and j["model"] == model]
    if not qualified:
        raise IntegrityError("Missing development assignments")
    for job in qualified:
        result = read(control / "jobs" / job["id"] / "result.json")
        if not result:
            return "waiting"
        if result["state"] != "complete":
            return "failed"
        verify_completion(Path(result["run"]), plan, job)
    gate = read(control / "gates" / (model + ".json"))
    if gate:
        if gate["plan_hash"] != plan["plan_hash"]:
            raise IntegrityError("Development gate identity mismatch")
        return "passed" if gate["all_pass"] else "failed"
    # Protocol capability, not clinical accuracy: evidence that this model can
    # actually retrieve, submit calculation code, and produce a final object.
    counts = collections.Counter()
    for job in qualified:
        result = read(control / "jobs" / job["id"] / "result.json")
        for file in (Path(result["run"]) / "episodes").glob("*.json.gz"):
            row = load_gzip(file)
            if row["split"] != "dev":
                raise IntegrityError("Test data in development gate")
            counts["episodes"] += 1
            counts["retrieval_episodes"] += row["raw_tool_calls"] > 0
            counts["code_episodes"] += row["code_calls"] > 0
            counts["final_episodes"] += row["stop"] == "final"
    passed = all(counts[k] > 0 for k in ("retrieval_episodes", "code_episodes", "final_episodes"))
    (control / "gates").mkdir(exist_ok=True)
    write_json(control / "gates" / (model + ".json"), {"all_pass": passed,
        "plan_hash": plan["plan_hash"], "counts": dict(counts), "accuracy_threshold": None,
        "criterion": "at least one retrieval, code call and final across complete development matrix",
        "time": time.time()})
    return "passed" if passed else "failed"


def eligible(control, plan, active, blocked):
    choices = []
    for model in MODELS:
        if model in blocked:
            continue
        key = "preflight--" + model
        folder = Path(control) / "preflight" / model
        if key not in active and not (folder / "result.json").exists():
            choices.append({"id": key, "model": model, "stage": "preflight"})
    for job in plan["jobs"]:
        if job["id"] in active or job["model"] in blocked:
            continue
        if (Path(control) / "jobs" / job["id"] / "result.json").exists():
            continue
        stress = read(Path(control) / "preflight" / job["model"] / "complete.json")
        if not stress or not stress.get("all_pass"):
            continue
        if job["stage"] == "qualification" or model_gate(control, plan, job["model"]) == "passed":
            choices.append(job)
    return choices


def execute(command, log, status, worker, fds, stop, timeout):
    output = Path(command[command.index("--out") + 1])
    start = last = time.monotonic()
    signature = None
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker["gpu"]),
        PYTHONPATH=str(ROOT / "repo"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with log.open("x") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
            env=environment, start_new_session=True, pass_fds=tuple(fds), cwd=ROOT)
        try:
            while child.poll() is None:
                progress = output / "progress.json"
                if progress.exists():
                    current = (progress.stat().st_mtime_ns, progress.stat().st_size)
                    if current != signature:
                        signature, last = current, time.monotonic()
                elapsed = time.monotonic() - start
                reason = ("controller_stop" if stop.is_set() else
                          "wall_timeout" if elapsed > timeout[1] else
                          "no_progress_timeout" if time.monotonic() - last > timeout[0] else None)
                write_json(status, {"state": "running", "pid": child.pid, "worker": worker,
                    "out": str(output), "seconds": elapsed, "heartbeat": time.time()})
                if reason:
                    stop_owned(child)
                    return {"returncode": child.returncode, "timeout": reason, "seconds": elapsed}
                time.sleep(5)
            return {"returncode": child.returncode, "timeout": None, "seconds": time.monotonic() - start}
        finally:
            stop_owned(child)


def attempt(control, job, worker, owner_fd, stop):
    control = Path(control)
    plan = load(control)
    pre = job["stage"] == "preflight"
    folder = control / "preflight" / job["model"] if pre else control / "jobs" / job["id"]
    with FileLease(control / "locks" / (job["id"] + ".lock")) as lease:
        if folder.exists():
            raise IntegrityError("Existing unresolved attempt: " + str(folder))
        folder.mkdir(parents=True)
        output = folder / "run" if pre else folder / "attempt1"
        command = [plan["python"], "-u", "-m", "medtoolbench.supplement.b_lite_execution",
            "preflight" if pre else "shard", "--control", str(control),
            "--worker", worker["name"], "--out", str(output)]
        command += ["--model", job["model"]] if pre else ["--job", job["id"]]
        write_json(folder / "command.json", {"command": command, "plan_hash": plan["plan_hash"],
            "worker": worker, "time": time.time(), "allocator": "expandable_segments:True"})
        execution = execute(command, folder / "run.log", control / "workers" / (worker["name"] + ".json"),
                            worker, [owner_fd, lease.fileno()], stop,
                            (plan["execution"]["stall_seconds"], plan["execution"]["wall_seconds"]))
        result = {"job": job["id"], "model": job["model"], "stage": job["stage"],
            "run": str(output), "worker": worker, "execution": execution,
            "plan_hash": plan["plan_hash"], "time": time.time()}
        completion = read(output / "complete.json")
        if execution["returncode"] == 0 and completion and completion.get("all_pass"):
            if pre:
                if completion["plan_hash"] != plan["plan_hash"]:
                    raise IntegrityError("Stress identity mismatch")
                write_json(folder / "complete.json", completion)
            else:
                # Re-run the original independent answer/source/accounting audit.
                from .external_runner import audit
                proof = audit(output, plan, job)
                if not proof["all_pass"]:
                    raise IntegrityError("Completed shard failed independent re-audit")
                verify_completion(output, plan, job)
            result.update(state="complete", episodes=0 if pre else len(job["episodes"]))
        else:
            if not pre and ((output / "audit.json").exists() or completion):
                raise IntegrityError("Audit-invalid evidence requires global review")
            failure = read(output / "failure.json") or {"error": str(execution)}
            result.update(state="quarantined", assigned_episodes=0 if pre else len(job["episodes"]),
                failure=failure, partial_rows_counted=False,
                failure_kind="cuda_oom" if "CUDA out of memory" in failure["error"] else "runtime_failure")
            if pre:
                write_json(folder / "failure.json", failure)
        write_json(folder / "result.json", result)
        write_json(control / "workers" / (worker["name"] + ".json"),
                   {"state": "idle", "job": job["id"], "result": result["state"], "heartbeat": time.time()})
        return result


def inventory(control, plan):
    done, failures, pending = [], [], []
    for job in plan["jobs"]:
        result = read(Path(control) / "jobs" / job["id"] / "result.json")
        if not result:
            pending.append(job)
        elif result["state"] == "complete":
            done.append((job, result))
        else:
            failures.append((job, result))
    return done, failures, pending


def snapshot(control, plan, active, blocked, started, baseline, fatal=None):
    done, failed, pending = inventory(control, plan)
    formal = sum(len(j["episodes"]) for j, _ in done if j["stage"] == "test")
    dev = sum(len(j["episodes"]) for j, _ in done if j["stage"] == "qualification")
    partial, running = 0, []
    for job, worker in active.values():
        pre = job["stage"] == "preflight"
        output = Path(control) / "preflight" / job["model"] / "run" if pre else Path(control) / "jobs" / job["id"] / "attempt1"
        progress = read(output / "progress.json") or {}
        if job["stage"] == "test":
            partial += progress.get("completed", 0)
        running.append({"job": job["id"], "model": job["model"], "stage": job["stage"],
                        "worker": worker["name"], "progress": progress})
    elapsed = time.time() - started
    rate = (formal + partial - baseline) * 3600 / elapsed if elapsed > 0 else 0
    total = sum(len(j["episodes"]) for j in plan["jobs"] if j["stage"] == "test")
    state = {"state": "draining_integrity_error" if fatal else "running", "heartbeat": time.time(),
        "plan_hash": plan["plan_hash"], "formal_total": total, "audited_formal": formal,
        "active_saved_not_audited": partial, "audited_development": dev,
        "active": running, "blocked_models": blocked, "fatal": fatal,
        "failures": [{"job": j["id"], "model": j["model"], "stage": j["stage"],
                      "assigned_episodes": len(j["episodes"]), "failure_kind": r.get("failure_kind")}
                     for j, r in failed], "pending_jobs": len(pending), "elapsed_seconds": elapsed,
        "observed_formal_per_hour": rate or None,
        "provisional_remaining_hours": (total - formal - partial) / rate if rate > 0 and not fatal and not blocked and not failed else None,
        "eta_note": "Includes development/loading; initial estimate biased. Per-model rates required for scheduling.",
        "failed_partial_rows_pooled": False}
    write_json(Path(control) / "status.json", state)
    return state


def report(control, plan):
    """Report a selected-cohort estimand with complete-matrix paired effects.

    Infrastructure missingness never becomes model failure or silently vanishes.
    Point effects/CIs require all assigned rows; partial reports show bounds.
    """
    control = Path(control)
    done, failures, pending = inventory(control, plan)
    rows, references = collections.defaultdict(dict), []
    for job, result in done:
        verify_completion(Path(result["run"]), plan, job)
        if job["stage"] != "test":
            continue
        expected = {e["key"] for e in job["episodes"]}
        seen = set()
        for file in (Path(result["run"]) / "episodes").glob("*.json.gz"):
            r = load_gzip(file)
            if r["key"] in seen or r["key"] in rows[job["model"]] or r["split"] != "test":
                raise IntegrityError("Duplicate row or development contamination")
            seen.add(r["key"])
            rows[job["model"]][r["key"]] = r
        if seen != expected:
            raise IntegrityError("Report coverage differs from shard")
        references.append({"job": job["id"], "audit_sha256": sha(Path(result["run"]) / "audit.json")})
    qs = [q for q in plan["questions"] if q["patient_group_split"] == "test"]
    summaries = {}
    lines = ["# B-lite external validation", "", "Selected, not previously attempted external-patient remainder; not a new hospital cohort.",
        "Original labels, full read-only agent workflow, fixed batch size 2. Historical B and development rows excluded.",
        "Effects average assigned hazards within question, questions within patient, then patients equally.",
        "No patient-harm, write-safety or population-prevalence claim. Infrastructure missingness is not agent failure.", ""]
    for model in MODELS:
        model_rows = rows[model]
        assigned = len(qs) * len(HAZARDS) * len(POLICIES)
        success = sum(bool(r["success"]) for r in model_rows.values())
        missing = assigned - len(model_rows)
        summary = {"assigned": assigned, "audited": len(model_rows), "missing": missing,
            "unweighted_success_bounds_all_conditions": [success / assigned, (success + missing) / assigned],
            "complete": not missing, "comparisons": {}, "conditions": {}}
        for policy in POLICIES:
            chosen = [r for r in model_rows.values() if r["condition"] == policy]
            clean = [r for r in chosen if r["hazard"] == "clean"]
            faults = [r for r in chosen if r["hazard"] != "clean"]
            summary["conditions"][policy] = {
                "assigned_clean": len(qs), "observed_clean": len(clean),
                "assigned_fault": len(qs) * 5, "observed_fault": len(faults),
                "correct_clean": sum(r["success"] for r in clean), "correct_fault": sum(r["success"] for r in faults),
                "mean_clean_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in clean) / len(clean) if clean else None,
                "mean_fault_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in faults) / len(faults) if faults else None,
                "mean_fault_raw_calls": sum(r["raw_tool_calls"] for r in faults) / len(faults) if faults else None,
                "stop_reasons": dict(collections.Counter(r["stop"] for r in chosen)),
                "by_hazard": {h: {"assigned": len(qs), "observed": sum(r["hazard"] == h for r in chosen),
                                   "correct": sum(r["success"] for r in chosen if r["hazard"] == h)} for h in HAZARDS}}
        if not missing:
            def get(q, h, c):
                return model_rows[q["question_id"] + "--" + h + "--" + c]
            for reference in ("vanilla", "informed"):
                for conditional in (False, True):
                    effects = collections.defaultdict(list)
                    for q in qs:
                        if conditional and not get(q, "clean", "vanilla")["success"]:
                            continue
                        delta = sum(int(get(q, h, "medguard")["success"]) - int(get(q, h, reference)["success"])
                                    for h in HAZARDS[1:]) / 5
                        effects[q["patient_fhir_id"]].append(delta)
                    key = "medguard-" + reference + ("/vanilla_clean_capable" if conditional else "")
                    summary["comparisons"][key] = patient_effect(effects)
            for policy in POLICIES:
                effects = collections.defaultdict(list)
                for q in qs:
                    delta = int(get(q, "clean", policy)["success"]) - sum(
                        int(get(q, h, policy)["success"]) for h in HAZARDS[1:]) / 5
                    effects[q["patient_fhir_id"]].append(delta)
                summary["conditions"][policy]["paired_clean_minus_fault"] = patient_effect(effects)
        summaries[model] = summary
        lines.extend([f"## {model}", "", f"Audited {len(model_rows)} / {assigned}; infrastructure-missing {missing}.", ""])
        if not missing:
            for key, value in summary["comparisons"].items():
                lines.append(f"- {key}: {value['difference']}; patient-cluster 95% interval {value['ci95']}.")
            lines.append("")
    output = control / "report"
    output.mkdir(exist_ok=True)
    full = not failures and not pending and all(s["complete"] for s in summaries.values())
    write_json(output / "metrics.json", {"plan_hash": plan["plan_hash"], "complete": full,
        "models": summaries, "statistics": plan["statistics"], "audit_references": references})
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return full


def run(control, resume=False):
    control = Path(control).resolve()
    plan = load(control)
    require_unit(control, plan)
    workers = plan["workers"]
    for worker in workers:
        validate_worker(plan, worker["name"])
    for name in ("workers", "preflight", "jobs", "locks", "sessions"):
        (control / name).mkdir(exist_ok=True)
    stop = threading.Event()
    def interrupted(signum, frame):
        stop.set()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with FileLease(control / "supervisor.lock") as owner:
        if (control / "complete.json").exists():
            raise IntegrityError("Campaign already complete")
        previous = read(control / "started.json")
        if previous and not resume:
            raise IntegrityError("Use explicit completed-shard reconciliation, not blind restart")
        # Compatible physical leases ensure no legacy/diagnostic model is alive.
        for worker in workers:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(worker["gpu"])
            with AllocatedGpuLease(worker):
                pass
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        done, failed, pending = inventory(control, plan)
        for job, result in done:
            verify_completion(Path(result["run"]), plan, job)
        for job in pending:
            if (control / "jobs" / job["id"]).exists():
                raise IntegrityError("Unresolved checkpoint needs explicit recovery: " + job["id"])
        baseline = sum(len(j["episodes"]) for j, _ in done if j["stage"] == "test")
        started = time.time()
        session = {"pid": os.getpid(), "time": started, "plan_hash": plan["plan_hash"],
                   "resume": resume, "prior_started": previous, "completed_formal_checkpoint": baseline}
        write_json(control / "sessions" / (str(time.time_ns()) + ".json"), session)
        if not previous:
            write_json(control / "started.json", session)
        blocked, busy, fatal = {}, {}, None
        for model in MODELS:
            if any(j["model"] == model and j["stage"] == "qualification" for j, _ in failed):
                blocked[model] = "qualification failure"
            if (control / "preflight" / model / "failure.json").exists():
                blocked[model] = "preflight failure"
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            while True:
                try:
                    load(control)
                    for future, (job, worker) in list(busy.items()):
                        if not future.done():
                            continue
                        del busy[future]
                        result = future.result()
                        if result["state"] != "complete" and job["stage"] in {"preflight", "qualification"}:
                            blocked[job["model"]] = job["stage"] + " failed"
                    _, failures, _ = inventory(control, plan)
                    counts = collections.Counter(j["model"] for j, r in failures)
                    for model, count in counts.items():
                        if count >= plan["execution"]["model_circuit_breaker_failures"]:
                            blocked[model] = "three failed shards; model circuit breaker"
                    for model in MODELS:
                        if model not in blocked and model_gate(control, plan, model) == "failed":
                            blocked[model] = "development gate failed"
                    if not fatal and not stop.is_set():
                        active_ids = {j["id"] for j, _ in busy.values()}
                        choices = eligible(control, plan, active_ids, blocked)
                        occupied = {w["name"] for _, w in busy.values()}
                        for worker in workers:
                            if worker["name"] not in occupied and choices:
                                job = choices.pop(0)
                                future = pool.submit(attempt, control, job, worker, owner.fileno(), stop)
                                busy[future] = (job, worker)
                except Exception:
                    fatal = traceback.format_exc()
                    write_json(control / "integrity_failure.json", {"error": fatal, "time": time.time()})
                state = snapshot(control, plan, busy, blocked, started, baseline, fatal)
                if not busy:
                    if fatal or stop.is_set() or not eligible(control, plan, set(), blocked):
                        break
                time.sleep(10)
        full = False if fatal or stop.is_set() else report(control, plan)
        if full:
            # Check historical outputs were not modified during the amendment.
            legacy = read(control / "legacy_inventory.json")
            changed = [p for p, checksum in legacy["files"].items() if not Path(p).is_file() or sha(p) != checksum]
            if changed:
                raise IntegrityError("Historical evidence changed: " + str(changed[:5]))
            write_json(control / "complete.json", {"all_pass": True, "plan_hash": plan["plan_hash"],
                "formal_episodes": state["formal_total"], "legacy_evidence_unchanged": True,
                "entire_project_complete": False, "time": time.time()})
        state.update(state="complete" if full else "needs_review", active=[], heartbeat=time.time())
        write_json(control / "status.json", state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("validate", "run", "preflight", "shard", "report"))
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--worker")
    parser.add_argument("--job")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "validate":
        validate(args.control)
    elif args.action == "preflight":
        preflight(args.control, args.model, args.worker, args.out)
    elif args.action == "shard":
        run_shard(args.control, args.job, args.worker, args.out)
    elif args.action == "report":
        report(args.control, load(args.control))
    else:
        try:
            run(args.control, args.resume)
        except BaseException:
            write_json(args.control / "controller_failure.json", {"error": traceback.format_exc(), "time": time.time()})
            raise


if __name__ == "__main__":
    main()
