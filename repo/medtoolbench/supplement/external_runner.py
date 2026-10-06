"""External QA evaluation on original labels and verified complete live histories."""
import argparse
import json
import os
import random
import signal
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.build import load_gzip, save_gzip
from medtoolbench.research.cluster_execution import AllocatedGpuLease, validate_worker
from medtoolbench.research.external_store import import_records, verified_roundtrip
from medtoolbench.research.fhir import HapiProcess
from medtoolbench.research.job_state import write_json
from medtoolbench.research.sql_reference import equivalent_rows
from .external_tools import KINDS, ExternalTools, Retrieval, messages
from .runner import SampleModel, load_plan


def parse(raw):
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    elif "<think>" in raw:
        raise ValueError("Incomplete reasoning block")
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    action, _ = json.JSONDecoder().raw_decode(raw)
    if not isinstance(action, dict):
        raise ValueError("Expected one action JSON object")
    if "final" in action:
        if set(action) != {"final"} or not isinstance(action["final"], dict):
            raise ValueError("Final cannot also contain a tool action")
    elif action.get("tool") not in {"metadata", "get_resources", "execute_python_code"}:
        raise ValueError("Unsupported read-only tool")
    return action


def correct(answer, expected):
    rows = answer.get("rows") if isinstance(answer, dict) else None
    if not isinstance(rows, list) or any(not isinstance(r, list) for r in rows):
        return False
    try:
        return equivalent_rows(rows, expected)
    except (TypeError, ValueError, OverflowError):
        return False


class Episode:
    def __init__(self, q, entry, resources, other, source_ref):
        self.q, self.entry = q, entry
        self.source_ref = source_ref
        self.retrieval = Retrieval(resources, q["patient_fhir_id"], entry["hazard"], other)
        self.tools = ExternalTools(self.retrieval, entry["condition"])
        self.messages = messages(q, entry["condition"])
        self.done, self.stop, self.answer, self.generations = False, "turn_limit", {}, []

    def advance(self, generation):
        self.generations.append(generation)
        self.messages.append({"role": "assistant", "content": generation["text"]})
        try:
            action = parse(generation["text"])
            if "final" in action:
                self.answer, self.done, self.stop = action["final"], True, "final"
                return
            response = self.tools.call(action)
        except (ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
            response = {"status": 400, "error": str(error)[:1000]}
        self.messages.append({"role": "user", "content": json.dumps(response, separators=(",", ":"))})

    def result(self):
        return {**self.entry, "question_id": self.q["question_id"], "patient": self.q["patient_fhir_id"],
            "group": self.q["patient_fhir_id"], "template": self.q["template"], "split": self.q["patient_group_split"],
            "answer": self.answer, "success": correct(self.answer, self.q["true_answer"]),
            "stop": self.stop, "messages": self.messages, "generations": self.generations,
            "events": self.tools.events, "retrieval_trace": self.retrieval.trace,
            "raw_tool_calls": self.tools.used, "triggered": self.retrieval.triggered,
            "code_calls": sum(e["tool"] == "execute_python_code" for e in self.tools.events),
            "input_tokens": sum(g["input_tokens"] for g in self.generations),
            "output_tokens": sum(g["output_tokens"] for g in self.generations),
            "source_ref": self.source_ref, "state_changing_tools": False, "inference_api": False}


def audit(output, plan, job):
    questions = {q["question_id"]: q for q in plan["questions"]}
    wanted, seen, errors = {e["key"] for e in job["episodes"]}, set(), []
    for p in (output / "episodes").glob("*.json.gz"):
        row = load_gzip(p)
        q = questions[row["question_id"]]
        issues = []
        if row["key"] in seen:
            issues.append("duplicate key")
        seen.add(row["key"])
        if correct(row["answer"], q["true_answer"]) != row["success"]:
            issues.append("independent answer regrade mismatch")
        if messages(q, row["condition"]) != row["messages"][:2]:
            issues.append("public prompt differs / possible label leakage")
        if row["triggered"] != sum(bool(t["injected"]) for t in row["retrieval_trace"]):
            issues.append("trigger count mismatch")
        if row["raw_tool_calls"] != len(row["retrieval_trace"]) or row["raw_tool_calls"] > 48:
            issues.append("tool budget mismatch")
        for name in ("input_tokens", "output_tokens"):
            if row[name] != sum(g[name] for g in row["generations"]):
                issues.append("token accounting mismatch")
        proof = json.loads((output / row["source_ref"]).read_text())
        if not proof["all_pass"] or proof["patient"] != q["patient_fhir_id"]:
            issues.append("unverified live source")
        if issues:
            errors.append({"key": row["key"], "issues": issues})
    report = {"all_pass": seen == wanted and not errors, "episodes": len(seen), "expected": len(wanted),
        "missing": sorted(wanted - seen), "unexpected": sorted(seen - wanted), "errors": errors, "plan_hash": plan["plan_hash"]}
    write_json(output / "audit.json", report)
    return report


def grouped(rows):
    return {kind: [r for r in rows if r["resourceType"] == kind] for kind in KINDS}


def main():
    p = argparse.ArgumentParser()
    for name in ("plan", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--worker", required=True)
    a = p.parse_args()
    plan = load_plan(a.plan)
    job = next(j for j in plan["jobs"] if j["id"] == a.job)
    worker = validate_worker(plan, a.worker)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker["gpu"])
    system = next(s for s in plan["systems"] if s["name"] == job["model"])
    if a.out.exists():
        raise RuntimeError("Never overwrite external experimental evidence")
    a.out.mkdir(parents=True)
    (a.out / "episodes").mkdir()
    (a.out / "sources").mkdir()
    write_json(a.out / "config.json", {"job": job, "plan_hash": plan["plan_hash"], "worker": worker, "inference_api": False})
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Received termination signal")
    signal.signal(signal.SIGTERM, interrupted)
    questions = {q["question_id"]: q for q in plan["questions"]}
    by_patient = defaultdict(list)
    for e in job["episodes"]:
        by_patient[questions[e["question"]]["patient_fhir_id"]].append(e)
    start, completed = time.monotonic(), 0
    try:
        with AllocatedGpuLease(worker), ThreadPoolExecutor(max_workers=system["batch_size"]) as pool:
            model = SampleModel(Path(system["path"]), system["thinking"], 20260917)
            write_json(a.out / "model.json", model.metadata)
            for patient, entries in sorted(by_patient.items()):
                record = plan["sources"][patient]
                resources = load_gzip(Path(record["path"]))
                if digest(resources) != record["hash"]:
                    raise RuntimeError("External original source changed")
                other_record = plan["sources"][record["other_patient"]]
                other_rows = load_gzip(Path(other_record["path"]))
                if digest(other_rows) != other_record["hash"]:
                    raise RuntimeError("External distractor source changed")
                other = grouped(other_rows)
                with HapiProcess(plan["image"], a.out / ("hapi-" + patient), heap="4g") as store:
                    store.types = KINDS
                    receipts = import_records(store, resources)
                    returned = verified_roundtrip(store, resources, receipts)
                    data = grouped(returned)
                    source_ref = "sources/" + patient + ".json"
                    write_json(a.out / source_ref, {"all_pass": True, "patient": patient, "source_hash": record["hash"],
                        "live_http_hash": digest(returned), "resources": len(returned), "receipt_hash": digest(receipts),
                        "source_path": record["path"], "other_patient": record["other_patient"], "all_history_verified": True})
                    random.Random(20260917).shuffle(entries)
                    for offset in range(0, len(entries), system["batch_size"]):
                        batch = entries[offset:offset + system["batch_size"]]
                        episodes = [Episode(questions[e["question"]], e, data, other, source_ref) for e in batch]
                        for turn in range(20):
                            active, prompts = [], []
                            for e in episodes:
                                if e.done:
                                    continue
                                prompt = model.prompt(e.messages)
                                if len(prompt) + 2048 > 16384:
                                    e.done, e.stop = True, "context_limit"
                                elif e.tools.used >= 48:
                                    e.done, e.stop = True, "raw_call_budget"
                                else:
                                    active.append(e)
                                    prompts.append(prompt)
                            if not active:
                                break
                            generations = model.generate(prompts, 2048)
                            list(pool.map(lambda pair: pair[0].advance(pair[1]), zip(active, generations)))
                        for e in episodes:
                            save_gzip(a.out / "episodes" / (e.entry["key"] + ".json.gz"), e.result())
                            completed += 1
                        progress = {"completed": completed, "total": len(job["episodes"]), "seconds": time.monotonic() - start, "heartbeat": time.time()}
                        write_json(a.out / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
                    verified_roundtrip(store, resources, receipts)
        proof = audit(a.out, plan, job)
        if not proof["all_pass"]:
            raise RuntimeError("External independent audit failed")
        write_json(a.out / "complete.json", {**progress, "all_pass": True, "inference_api": False})
    except BaseException:
        write_json(a.out / "failure.json", {"error": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
