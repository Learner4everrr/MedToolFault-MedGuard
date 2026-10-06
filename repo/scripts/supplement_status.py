"""Read-only multi-campaign status; qualification is never reported as test data."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from medtoolbench.supplement.runner import load_plan
from medtoolbench.supplement.gptoss_native_status import snapshot as native_snapshot

p = argparse.ArgumentParser()
p.add_argument("--pipeline", type=Path, default=Path("data/supplement_pipeline_8gpu_v2/pipeline.json"))
p.add_argument("--brief", action="store_true", help="Omit detailed worker paths and command records")
p.add_argument("--memory-manifest", type=Path, default=Path("data/supplement_memory_recovery_v1/manifest.json"))
p.add_argument("--isolated-manifest", type=Path, default=Path("data/supplement_isolated_execution_v1/manifest.json"))
p.add_argument("--storage-manifest", type=Path, default=Path("data/supplement_storage_recovery_v1/manifest.json"))
a = p.parse_args()
# An explicit model migration supersedes the old all-Gemma-inclusive progress
# denominator. Report it separately; never count retired Gemma as GPT-OSS.
migration_root = Path('data/gptoss_replacement_v1')
protocol_root = Path('data/gptoss_protocol_repair_v2')
if (protocol_root / 'handoff.json').exists():
    migration_root = protocol_root
migration = None
if (migration_root / 'handoff.json').exists():
    def migration_record(name):
        file = migration_root / name
        return json.loads(file.read_text()) if file.exists() else None
    m = migration_record('manifest.json')
    bootstraps = [json.loads(p.read_text()) for p in migration_root.glob('bootstrap*.json')]
    bootstrap = max(bootstraps, key=lambda r: r.get('time', 0), default=None)
    summaries = []
    if m:
        for name in m['old_campaigns'] + m['new_campaigns']:
            cp = Path(name)
            plan = load_plan(cp)
            counts = {stage: {'expected': 0, 'audited': 0, 'saved_partial': 0}
                      for stage in ('qualification', 'test')}
            for job in plan['jobs']:
                if job['model'] in m['retired_models']:
                    continue
                c = counts[job['stage']]
                c['expected'] += len(job['episodes'])
                folder = cp.parent / 'jobs' / job['id']
                result = json.loads((folder / 'result.json').read_text()) if (folder / 'result.json').exists() else None
                if result and result.get('state') == 'complete':
                    c['audited'] += result['episodes']
                elif not result:
                    attempts = sorted(folder.glob('attempt[0-9]*'))
                    if attempts and not (attempts[-1] / 'failure.json').exists() and (attempts[-1] / 'progress.json').exists():
                        c['saved_partial'] += json.loads((attempts[-1] / 'progress.json').read_text())['completed']
            summaries.append({'plan': str(cp), 'counts': counts,
                'cohort': 'GPT-OSS fresh matrix' if name in m['new_campaigns'] else 'unchanged non-Gemma controls'})
    migration = {'state': migration_record('status.json') or bootstrap or {'state': 'preparing'},
        'control': str(migration_root),
        'protocol_revision': 'harmony-json-bridge-v2' if migration_root == protocol_root else 'v1-final-only',
        'old_protocol_invalid_output_excluded': migration_root == protocol_root,
        'bootstrap': bootstrap,
        'manifest': str(migration_root / 'manifest.json') if m else None,
        'retired_models': ['gemma3_12b'], 'campaigns': summaries,
        'complete': migration_record('complete.json'), 'needs_review': migration_record('needs_review.json')}
priority = None
priority_root = Path('data/non_gptoss_priority_v1')
for candidate in ('data/non_gptoss_priority_v2', 'data/non_gptoss_priority_v3', 'data/acd_priority_order_v1',
                  'data/acd_llama_expansion_v1'):
    if (Path(candidate) / 'manifest.json').exists():
        priority_root = Path(candidate)
if (priority_root / 'manifest.json').exists():
    def priority_record(name):
        file = priority_root / name
        return json.loads(file.read_text()) if file.exists() else None
    pm = priority_record('manifest.json')
    counts = {stage: {'expected': 0, 'audited': 0, 'saved_partial': 0}
              for stage in ('qualification', 'test')}
    by_model = {}
    by_campaign = {}
    full_matrix_by_model = {}
    for raw in pm['old_campaigns'] + pm.get('new_campaigns', []):
        cp = Path(raw)
        plan = load_plan(cp)
        for job in plan['jobs']:
            if job['model'] not in pm['allowed_models']:
                continue
            c = counts[job['stage']]
            c['expected'] += len(job['episodes'])
            cc = by_campaign.setdefault(cp.parent.name, {
                stage: {'expected': 0, 'audited': 0, 'saved_partial': 0}
                for stage in ('qualification', 'test')})[job['stage']]
            cc['expected'] += len(job['episodes'])
            key = str(cp.resolve()) + '::' + job['id']
            repaired = key in pm.get('repair_jobs', {})
            repair_root = (Path(pm['repair_control']) if pm.get('repair_control') else
                           Path(pm['scheduling_parent']).parent if pm.get('scheduling_parent') else priority_root)
            f = (repair_root / 'repair_jobs' / job['id']) if repaired else (cp.parent / 'jobs' / job['id'])
            result = json.loads((f / 'result.json').read_text()) if (f / 'result.json').exists() else None
            model = by_model.setdefault(job['model'], {'formal_expected': 0, 'formal_audited': 0})
            if job['stage'] == 'test':
                model['formal_expected'] += len(job['episodes'])
            full_coverage = (job['stage'] == 'test' and job['model'] in pm.get('full_matrix_models', [])
                             and str(cp) in pm.get('campaign_roles', {})
                             and job.get('arm') != 'adverse_model_actions')
            if full_coverage:
                full_model = full_matrix_by_model.setdefault(job['model'],
                    {'formal_expected': 0, 'formal_audited': 0, 'saved_partial': 0})
                full_model['formal_expected'] += len(job['episodes'])
            if result and result.get('state') == 'complete':
                c['audited'] += result['episodes']
                cc['audited'] += result['episodes']
                if job['stage'] == 'test':
                    model['formal_audited'] += result['episodes']
                if full_coverage:
                    full_model['formal_audited'] += result['episodes']
            elif not result:
                attempts = sorted(p for p in f.glob('attempt[0-9]*') if p.is_dir())
                if attempts and not (attempts[-1] / 'failure.json').exists():
                    progress = attempts[-1] / 'progress.json'
                    if progress.exists():
                        saved = json.loads(progress.read_text())['completed']
                        c['saved_partial'] += saved
                        cc['saved_partial'] += saved
                        if full_coverage:
                            full_model['saved_partial'] += saved
    dispatch_names = ({Path(p).parent.name for p in pm['campaign_roles']} if pm.get('campaign_roles')
                      else set(pm.get('dispatch_campaigns', by_campaign)))
    priority_counts = {stage: {key: sum(c[stage][key] for name, c in by_campaign.items() if name in dispatch_names)
                      for key in ('expected', 'audited', 'saved_partial')}
                      for stage in ('qualification', 'test')}
    priority = {'manifest': str(priority_root / 'manifest.json'), 'state': priority_record('status.json'),
        'allowed_models': pm['allowed_models'], 'deferred_models': pm['deferred_models'],
        'automatic_deferred_resume': False, 'counts': counts, 'by_model': by_model,
        'by_campaign': by_campaign, 'dispatch_campaigns': pm.get('dispatch_campaigns'),
        'automatic_b_resume': pm.get('automatic_b_resume'),
        'new_models_added': pm.get('new_models_added', []),
        'full_matrix_models': pm.get('full_matrix_models'),
        'full_matrix_by_model': full_matrix_by_model,
        'priority_counts_excluding_b': priority_counts,
        'campaign_roles': pm.get('campaign_roles'),
        'complete': priority_record('complete.json'), 'needs_review': priority_record('needs_review.json'),
        'repair_jobs': len(pm.get('repair_jobs', {})),
        'repair_formal_episodes': sum(r['episodes'] for r in pm.get('repair_jobs', {}).values()),
        'original_failed_evidence_preserved': pm.get('original_failed_records_immutable',
                                                     pm.get('original_scientific_plans_unchanged', False))}
    ps = priority['state'] or {}
    ph = priority_record('handoff.json') or {}
    active = [priority_record('workers/' + name + '.json') for name in ps.get('busy', {})]
    active += [r for r in ph.get('live_children_preserved', []) if r['pid'] in ps.get('preserved_children', [])]
    priority['active_workers'] = []
    for worker in active:
        if not worker:
            continue
        out = Path(worker['out'])
        progress = json.loads((out / 'progress.json').read_text()) if (out / 'progress.json').exists() else None
        priority['active_workers'].append({'worker': worker['worker']['name'], 'pid': worker['pid'],
            'out': str(out), 'progress': progress,
            'inherited_from_previous_controller': worker['pid'] in ps.get('preserved_children', [])})
    if migration:
        migration['historical_state_before_deferral'] = migration['state']
        migration['state'] = {'state': 'deferred_by_user', 'not_current_dispatcher': True}
config = json.loads(a.pipeline.read_text())
pause_marker = a.pipeline.parent / "paused.json"
pause = json.loads(pause_marker.read_text()) if pause_marker.exists() else None
reports = []
for name in config["campaigns"]:
    path = Path(name)
    plan = load_plan(path)
    counts = {"qualification": {"saved": 0, "audited": 0, "expected": 0}, "test": {"saved": 0, "audited": 0, "expected": 0}}
    failed, completed_jobs = [], 0
    for job in plan["jobs"]:
        kind = job["stage"]
        counts[kind]["expected"] += len(job["episodes"])
        folder = path.parent / "jobs" / job["id"]
        result = json.loads((folder / "result.json").read_text()) if (folder / "result.json").exists() else None
        if result and result["state"] == "complete":
            counts[kind]["saved"] += result["episodes"]
            counts[kind]["audited"] += result["episodes"]
            completed_jobs += 1
        else:
            if result:
                failed.append(job["id"])
            attempts = sorted((x for x in folder.glob("attempt*") if x.is_dir() and x.name[7:].isdigit()),
                              key=lambda x: int(x.name[7:]))
            # Partial rows of a terminal failed attempt are preserved evidence,
            # not current scientific progress (whole-shard recovery restarts).
            if not result and attempts and not (attempts[-1] / "failure.json").exists() and (attempts[-1] / "progress.json").exists():
                counts[kind]["saved"] += json.loads((attempts[-1] / "progress.json").read_text())["completed"]
    oracle = path.parent / "oracle"
    orow = None
    for filename in ("failed.json", "complete.json", "progress.json"):
        if (oracle / filename).exists():
            content = json.loads((oracle / filename).read_text())
            orow = {"marker": filename, **{k: v for k, v in content.items() if k not in {"results", "trace"}}}
            break
    workers = {x.stem: json.loads(x.read_text()) for x in (path.parent / "workers").glob("*.json")}
    reports.append({"campaign": path.parent.name, "plan_hash": plan["plan_hash"], "counts": counts,
        "jobs_complete": completed_jobs, "jobs_total": len(plan["jobs"]), "failed_jobs": failed,
        "oracle": orow, "workers": workers, "run_complete": (path.parent / "complete.json").exists(),
        "report_complete": (path.parent / "report/complete.json").exists()})
if a.brief:
    for report in reports:
        workers = report.pop("workers")
        report["worker_states"] = dict(Counter(w["state"] for w in workers.values()))
        report["oldest_worker_heartbeat_seconds"] = round(max((time.time() - w["heartbeat"] for w in workers.values()), default=0), 1)
continuation = {p.stem: json.loads(p.read_text()).get("state") for p in a.pipeline.parent.glob("*-gpu*.json")}
memory = None
if a.memory_manifest.exists():
    manifest = json.loads(a.memory_manifest.read_text())
    if Path(manifest["canonical_pipeline"]).resolve() == a.pipeline.resolve():
        root = a.memory_manifest.parent
        def record(name):
            path = root / name
            return json.loads(path.read_text()) if path.exists() else None
        preflight, development = record("preflight/complete.json"), record("qualification/complete.json")
        progress = record("preflight/progress.json")
        state = record("status.json")
        pending, historical = [], []
        for cp in manifest["campaigns"]:
            for authorization in (Path(cp).parent / "jobs").glob("*/memory_repair_authorization.json"):
                result = authorization.parent / "result.json"
                repaired = result.exists() and json.loads(result.read_text())["state"] == "complete"
                item = {"campaign": Path(cp).parent.name, "job": authorization.parent.name, "repaired_and_audited": repaired}
                historical.append(item)
                if not repaired:
                    pending.append(item)
        active_workers = {}
        for name in (state or {}).get("busy", {}):
            worker = record("workers/" + name + ".json")
            if worker:
                active_workers[name] = worker
        memory = {"manifest": str(a.memory_manifest), "state": state, "failure": record("failed.json"),
            "superseded": record("superseded.json"),
            "numeric_preflight_passed": bool(preflight and preflight["all_pass"]),
            "numeric_preflight_last_check": (progress or {}).get("checks", [None])[-1],
            "development_regression": development, "unresolved_oom_repairs": pending,
            "historical_oom_repairs": historical, "active_workers": active_workers,
            "complete": record("complete.json"), "extra_validation_excluded_from_test_counts": True}
        if a.brief:
            memory["active_worker_states"] = dict(Counter(w["state"] for w in active_workers.values()))
            memory.pop("active_workers")
isolated = None
if a.isolated_manifest.exists():
    manifest = json.loads(a.isolated_manifest.read_text())
    if Path(manifest["canonical_pipeline"]).resolve() == a.pipeline.resolve():
        root = a.isolated_manifest.parent
        def isolated_record(name):
            path = root / name
            return json.loads(path.read_text()) if path.exists() else None
        state = isolated_record("status.json")
        active = {}
        for name, assignment in (state or {}).get("busy", {}).items():
            worker = isolated_record("workers/" + name + ".json")
            if worker:
                out = Path(worker["out"]) if worker.get("out") else None
                progress = json.loads((out / "progress.json").read_text()) if out and (out / "progress.json").exists() else None
                active[name] = {"assignment": assignment, "state": worker["state"],
                    "pid": worker.get("pid"), "out": worker.get("out"),
                    "heartbeat_age_seconds": round(time.time() - worker.get("heartbeat", 0), 1),
                    "progress": ({k: progress[k] for k in ("completed", "total", "seconds", "heartbeat") if k in progress}
                                 if progress else None)}
        dev_root = root / "gemma_development"
        dev_results = [json.loads(p.read_text()) for p in dev_root.glob("*/result.json")]
        isolated = {"manifest": str(a.isolated_manifest), "state": state,
            "controller_heartbeat_age_seconds": round(time.time() - state["heartbeat"], 1) if state else None,
            "legacy_controllers_superseded": bool(isolated_record("handoff.json")),
            "active_workers": active, "unit_regression": isolated_record("validation/unit.json"),
            "real_hapi_regression": isolated_record("validation/hapi.json"),
            "preflight": {mode: {"passed": bool(isolated_record("preflight/" + mode + "/complete.json")),
                "failure": isolated_record("preflight/" + mode + "/failed.json")}
                for mode in ("compare", "stress")},
            "gemma_extra_development_audited": sum(r.get("episodes", 0) for r in dev_results if r["state"] == "complete"),
            "gemma_extra_development_complete": isolated_record("gemma_development/complete.json"),
            "needs_review": isolated_record("needs_review.json"), "complete": isolated_record("complete.json"),
            "extra_validation_excluded_from_test_counts": True}
        for report in reports:
            report["legacy_workers_superseded"] = isolated["legacy_controllers_superseded"]
        if a.brief and isolated["real_hapi_regression"]:
            isolated["real_hapi_regression"].pop("checks", None)
storage = None
if a.storage_manifest.exists():
    manifest = json.loads(a.storage_manifest.read_text())
    if Path(manifest["canonical_pipeline"]).resolve() == a.pipeline.resolve():
        root = a.storage_manifest.parent
        def storage_record(name):
            path = root / name
            return json.loads(path.read_text()) if path.exists() else None
        state = storage_record("status.json")
        active = {}
        for name, assignment in (state or {}).get("busy", {}).items():
            worker = storage_record("workers/" + name + ".json")
            if worker:
                out = Path(worker["out"]) if worker.get("out") else None
                progress = json.loads((out / "progress.json").read_text()) if out and (out / "progress.json").exists() else None
                active[name] = {"assignment": assignment, "state": worker["state"],
                    "pid": worker.get("pid"), "out": worker.get("out"),
                    "heartbeat_age_seconds": round(time.time() - worker.get("heartbeat", 0), 1),
                    "progress": ({k: progress[k] for k in ("completed", "total", "seconds", "heartbeat") if k in progress}
                                 if progress else None)}
        reconciliation = storage_record("reconciliation.json") or {}
        storage = {"manifest": str(a.storage_manifest), "state": state,
            "superseded_by_gptoss_migration": migration is not None,
            "controller_heartbeat_age_seconds": round(time.time() - state["heartbeat"], 1) if state else None,
            "active_workers": active, "unit_regression": storage_record("unit_tests.json"),
            "completed_jobs_retained": len(reconciliation.get("completed", [])),
            "quota_interrupted_jobs": len(reconciliation.get("interrupted", [])),
            "historical_partial_rows_preserved_not_pooled": sum(r["saved_episodes"] for r in reconciliation.get("interrupted", [])),
            "needs_review": storage_record("needs_review.json"), "complete": storage_record("complete.json")}
        if isolated:
            isolated["superseded_by_storage_recovery"] = bool(storage_record("started.json"))
print(json.dumps({"time": time.time(), "pipeline": str(a.pipeline), "pause": pause,
    "gptoss_native_acd": native_snapshot() if Path('data/gptoss_acd_native_v3').exists() else None,
    "active_other_model_priority": priority,
    "gptoss_replacement": migration,
    "historical_campaign_counts_below_include_retired_gemma": migration is not None,
    "predecessor_pipeline": config.get("predecessor_pipeline"), "continuation_states": dict(Counter(continuation.values())),
    "memory_execution": memory,
    "isolated_execution": isolated,
    "storage_recovery": storage,
    "expected_episodes_including_qualification": config["episodes"],
    "campaigns": reports}, indent=2))
