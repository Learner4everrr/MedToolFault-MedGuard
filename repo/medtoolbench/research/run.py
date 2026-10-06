"""Batched local LLM interaction with one owned real HAPI per active episode."""
import argparse
import json
import random
import signal
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from medtoolbench.storage import digest
from .build import load_fixture, save_gzip
from .contracts import grade
from .fhir import HapiProcess, normalized_state
from .models import LocalModel
from .protocol import SYSTEM, messages, parse_action
from .transport import Client, Proxy
from .job_state import write_json
from .runtime import GpuLease, core_fingerprint, recover_batches


class Episode:
    def __init__(self, store, task, hazard, condition, benchmark):
        self.store, self.task, self.hazard, self.condition = store, task, hazard, condition
        self.key = f"{task['id']}--{hazard}--{condition}"
        fixture = load_fixture(benchmark, task)
        if digest(fixture) != task["fixture_hash"]:
            raise RuntimeError("Fixture hash mismatch")
        self.before = store.reload(fixture)
        self.proxy = Proxy(store, task, hazard)
        self.client = Client(self.proxy.call, condition)
        self.messages = messages(task, condition, hazard)
        self.answer, self.done, self.stop = {}, False, "turn_limit"
        self.generations, self.parsed_actions = [], []

    def advance(self, generation):
        self.generations.append(generation)
        raw = generation["text"]
        self.messages.append({"role": "assistant", "content": raw})
        try:
            action = parse_action(raw)
            self.parsed_actions.append(action)
            if "final" in action:
                self.answer, self.done, self.stop = action["final"], True, "final"
                return
            result = self.client.call(action["method"], action["path"], action.get("body"))
        except (ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
            result = {"status": 400, "body": {"error": str(error), "instruction": "Reply with one valid action JSON object."}}
        self.messages.append({"role": "user", "content": json.dumps(result, separators=(",", ":"))})

    def result(self):
        after = self.store.snapshot()
        g = grade(self.task, self.before, after, self.answer)
        transient = [{"trace_index": r["trace_index"], "grade": grade(self.task, self.before, r["state"], {})}
                     for r in self.proxy.mutation_states]
        # Missing later-required steps are not unsafe mutations. Only the unsafe
        # list is used for transient safety, never partial-workflow success.
        return {"key": self.key, "task_id": self.task["id"], "group": self.task["group"],
                "family": self.task["family"], "template": self.task["template"], "split": self.task["split"],
                "patient": self.task["patient"], "hazard": self.hazard, "condition": self.condition,
                "action_required": bool(self.task["contract"]["required_creations"]),
                "answer": self.answer, "stop": self.stop, "grade": g,
                "before": self.before, "after_semantic_hash": digest(normalized_state(after)),
                "trace": self.proxy.trace, "triggered": self.proxy.triggered,
                "transient_states": transient, "any_transient_unsafe": any(r["grade"]["unsafe"] for r in transient),
                "messages": self.messages, "generations": self.generations, "parsed_actions": self.parsed_actions,
                "tool_calls": len(self.proxy.trace),
                "verification_or_retry_calls": sum(r["actor"] != "agent" for r in self.proxy.trace),
                "input_tokens": sum(r["input_tokens"] for r in self.generations),
                "output_tokens": sum(r["output_tokens"] for r in self.generations),
                "inference_seconds_amortized": sum(r["batch_seconds_per_item"] for r in self.generations),
                "tool_seconds_including_audit": sum(r["wall_seconds"] for r in self.proxy.trace)}


def selected_tasks(manifest, split, per_template):
    tasks = [t for t in manifest if t["split"] == split]
    if per_template:
        counts, selected = Counter(), []
        for t in tasks:
            if counts[t["template"]] < per_template:
                counts[t["template"]] += 1
                selected.append(t)
        tasks = selected
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--per-template", type=int)
    p.add_argument("--conditions", nargs="+", choices=["vanilla", "warning", "retry_only", "medguard", "oracle_hint"], default=["vanilla", "medguard"])
    p.add_argument("--hazards", nargs="+")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--max-context", type=int, default=16384)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--task-shards", type=int, default=1)
    p.add_argument("--task-shard", type=int, default=0)
    p.add_argument("--resume-from", type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Run directory exists; never overwrite experimental evidence")
    if not 1 <= a.batch_size <= 8:
        raise SystemExit("Use 1–8 isolated workers")
    if not 0 <= a.task_shard < a.task_shards:
        raise SystemExit("Invalid task shard")
    if len(set(a.conditions)) != len(a.conditions):
        raise SystemExit("Conditions must not repeat")
    manifest = json.loads((a.benchmark / "tasks.json").read_text())
    tasks = selected_tasks(manifest, a.split, a.per_template)[a.task_shard::a.task_shards]
    validation = json.loads((a.validation / "complete.json").read_text())
    vconfig = json.loads((a.validation / "config.json").read_text())
    if not validation["all_valid"] or digest(manifest) != vconfig["task_manifest_hash"]:
        raise SystemExit("Require completed, matching clean/recovery oracle validation")
    if not tasks or not {t["id"] for t in tasks} <= set(vconfig["selected_task_ids"]):
        raise SystemExit("Selected tasks have not all passed real HAPI oracle validation")
    jobs = [(t, h, c) for t in tasks for h in t["hazards"] for c in a.conditions if not a.hazards or h in a.hazards]
    if not jobs:
        raise SystemExit("Empty run matrix")
    random.Random(a.seed).shuffle(jobs)
    a.out.mkdir(parents=True)
    (a.out / "episodes").mkdir()
    (a.out / "states").mkdir()
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(manifest_hash=digest(manifest), expected_keys=[f"{t['id']}--{h}--{c}" for t, h, c in jobs],
                  core_hashes=core_fingerprint(),
                  started_at=datetime.now(timezone.utc).isoformat(), system_prompt=SYSTEM,
                  source_hashes={str(f.relative_to(Path(__file__).parent)): digest(f.read_text()) for f in Path(__file__).parent.glob("*.py")},
                  inference_api=False, backend="real HAPI, isolated process per batch slot",
                  latency_method="Batched inference time amortized per active episode; not serial latency")
    write_json(a.out / "config.json", config)
    begin, summaries = time.perf_counter(), []
    batches = [jobs[i:i+a.batch_size] for i in range(0, len(jobs), a.batch_size)]
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received termination signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if a.resume_from:
            summaries, batches, recovery = recover_batches(a.resume_from, a.out, a.benchmark, config, batches)
            write_json(a.out / "recovery.json", recovery)
        with ExitStack() as stack, ThreadPoolExecutor(max_workers=a.batch_size) as pool:
            stack.enter_context(GpuLease())
            model = LocalModel(a.model, a.thinking, a.seed)
            write_json(a.out / "model.json", model.metadata)
            owners = [HapiProcess(a.image, a.out / f"hapi-{i}", heap="2g") for i in range(a.batch_size)]
            for owner in owners:
                stack.callback(owner.close)
            stores = list(pool.map(lambda o: o.start(), owners))
            for batch in batches:
                episodes = list(pool.map(lambda x: Episode(stores[x[0]], *x[1], a.benchmark), enumerate(batch)))
                for turn in range(a.max_turns):
                    active = [e for e in episodes if not e.done]
                    if not active:
                        break
                    ready, prompts = [], []
                    for e in active:
                        prompt = model.prompt(e.messages)
                        length = len(prompt)
                        if length + a.max_new_tokens > a.max_context:
                            e.done, e.stop = True, "context_limit"
                        else:
                            ready.append(e)
                            prompts.append(prompt)
                    if not ready:
                        continue
                    generations = model.generate(prompts, a.max_new_tokens)
                    list(pool.map(lambda pair: pair[0].advance(pair[1]), zip(ready, generations)))
                for row in pool.map(lambda e: e.result(), episodes):
                    before = row.pop("before")
                    before_hash = digest(before)
                    state_path = a.out / "states" / (before_hash + ".json.gz")
                    if not state_path.exists():
                        save_gzip(state_path, before)
                    row["before_ref"] = str(state_path.relative_to(a.out))
                    row["total_tokens"] = row["input_tokens"] + row["output_tokens"]
                    save_gzip(a.out / "episodes" / (row["key"] + ".json.gz"), row)
                    summaries.append({"key": row["key"], "success": row["grade"]["success"],
                                      "unsafe": row["grade"]["unsafe"], "triggered": row["triggered"], "stop": row["stop"]})
                progress = {"completed": len(summaries), "total": len(jobs), "success": sum(r["success"] for r in summaries),
                            "seconds": round(time.perf_counter() - begin, 1)}
                write_json(a.out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        assert {r["key"] for r in summaries} == set(config["expected_keys"])
        progress = {"completed": len(summaries), "total": len(jobs), "success": sum(r["success"] for r in summaries),
                    "seconds": round(time.perf_counter() - begin, 1)}
        write_json(a.out / "complete.json", {**progress, "finished_at": datetime.now(timezone.utc).isoformat(),
            "inference_api": False, "all_expected_present": True})
    except BaseException:
        (a.out / "infrastructure_failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
