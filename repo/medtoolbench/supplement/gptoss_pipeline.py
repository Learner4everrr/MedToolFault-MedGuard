"""Explicit model retirement, unchanged-control continuation, new GPT-OSS matrix.

One successor owns eight physical GPU lanes. Existing non-Gemma children finish
naturally, then are audited/adopted, never rerun merely because their dispatcher
was retired. Original scientific plans and successful evidence remain immutable.
"""
import argparse
import copy
import json
import os
import subprocess
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import AllocatedGpuLease, FileLease, validate_worker
from .isolated_execution import model_gate
from .memory_execution import verify_completion
from .runner import load_plan
from .storage_recovery import Journal, atomic_json, execute, read

RETIRED = 'gemma3_12b'
TARGET = 'gptoss20b'


def process_cmdline(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return ''


def source_hashes(root):
    files = [root / 'repo/medtoolbench/supplement' / (n + '.py') for n in
        ('gptoss_model', 'gptoss_mask', 'gptoss_runner', 'gptoss_pipeline', 'input_guard')]
    files += [root / 'repo/scripts' / n for n in ('gptoss_smoke.py', 'gptoss_artifacts.py', 'validate_gptoss.py', 'validate_gptoss_mask.py')]
    files += [root / 'repo/tests/test_gptoss_replacement.py']
    return {str(p): digest(p.read_text()) for p in files}


def replacement_plan(parent, parent_path, root, control, smoke, sources, unit_proof=None):
    from .gptoss_model import REVISION, REASONING
    plan = copy.deepcopy(parent)
    plan.pop('plan_hash')
    original = [j for j in parent['jobs'] if j['model'] == RETIRED]
    plan['jobs'] = [dict(copy.deepcopy(j), id=j['id'].replace(RETIRED, TARGET), model=TARGET) for j in original]
    previous = next(s for s in parent['systems'] if s['name'] == RETIRED)
    plan['systems'] = [{'name': TARGET, 'repo_id': 'openai/gpt-oss-20b', 'family': 'GPT-OSS',
        'thinking': True, 'reasoning_effort': REASONING, 'batch_size': previous['batch_size'],
        'path': str(root / 'data/models/gpt-oss-20b'), 'revision': REVISION, 'model_type': 'gpt_oss',
        'quantization': 'native MXFP4', 'inference_api': False}]
    if previous['batch_size'] != 2:
        raise RuntimeError('Unexpected original logical batch size')
    plan['python'] = str(root / 'env_gptoss/bin/python')
    plan['session'] = 'medtoolbench-gptoss-v1'
    plan['created_at'] = time.time()
    plan['source_hashes'].update(sources)
    inputs = [parent_path, smoke, control / 'artifacts.json', unit_proof or control / 'unit_tests_final.json',
              control / 'mask_equivalence_v1/complete.json']
    oracle = parent_path.parent / 'oracle/complete.json'
    if parent.get('require_oracle'):
        proof = read(oracle)
        if not proof or not proof.get('all_pass') or proof.get('plan_hash') != parent['plan_hash']:
            raise RuntimeError('Parent environment oracle did not pass')
        inputs.append(oracle)
    plan['require_oracle'] = False
    plan['gptoss_replacement'] = {'parent_plan': str(parent_path), 'parent_plan_hash': parent['plan_hash'],
        'retired_model': RETIRED, 'target_model': TARGET, 'smoke': str(smoke),
        'parent_oracle_reused': str(oracle) if parent.get('require_oracle') else None,
        'oracle_rationale': 'Same tasks, fixtures, faults and graders; certificate tests environment, not the new model',
        'scope': 'Full former Gemma matrix, including jobs Gemma previously completed; fresh outputs only',
        'selection': 'User compute-driven amendment after primary results; exploratory, not preregistered',
        'reasoning_effort': REASONING, 'native_quantization': 'MXFP4', 'physical_microbatch': 2,
        'clinical_protocol_changed': False, 'token_budget_includes_reasoning': True}
    for file in inputs:
        plan['input_hashes'][str(file)] = digest(file.read_bytes().hex())
    for old, new in zip(original, plan['jobs']):
        if {k: v for k, v in old.items() if k not in ('id', 'model')} != {k: v for k, v in new.items() if k not in ('id', 'model')}:
            raise RuntimeError('Replacement changed a task/condition/budget/seed assignment')
    plan['plan_hash'] = digest(plan)
    return plan


def prepare(control, smoke, unit_proof):
    root = Path(__file__).resolve().parents[3]
    manifest_path = control / 'manifest.json'
    if manifest_path.exists():
        raise RuntimeError('Never overwrite a frozen migration')
    sources = source_hashes(root)
    proof = read(smoke)
    if proof and proof.get('diagnostic_only'):
        raise RuntimeError('Diagnostic CUDA runs cannot authorize formal dispatch')
    if not proof or not proof['all_pass'] or proof['adapter_hash'] != sources[str(Path(__file__).with_name('gptoss_model.py'))]:
        raise RuntimeError('Passing exact-adapter GPU preflight required')
    mask_proof = read(control / 'mask_equivalence_v1/complete.json')
    if not mask_proof or not mask_proof.get('all_pass') or mask_proof.get('repair_hash') != sources[str(Path(__file__).with_name('gptoss_mask.py'))]:
        raise RuntimeError('Passing exact-repair GPU mask equivalence required')
    tests = read(unit_proof)
    if not tests or not tests.get('all_pass') or tests.get('source_hashes') != sources:
        raise RuntimeError('Passing exact-source unit tests required')
    if not read(control / 'artifacts.json').get('all_pass'):
        raise RuntimeError('Pinned artifacts required')
    old = read(root / 'data/supplement_isolated_execution_v1/manifest.json')
    old_paths, new_paths, hashes = [], [], {}
    for index, raw in enumerate(old['campaigns']):
        parent_path = Path(raw)
        parent = load_plan(parent_path)
        path = control / ('campaign_' + 'abcd'[index]) / 'plan.json'
        if path.exists():
            raise RuntimeError('Partial preparation exists; investigate it')
        plan = replacement_plan(parent, parent_path, root, control, smoke, sources, unit_proof)
        atomic_json(path, plan)
        old_paths.append(str(parent_path))
        new_paths.append(str(path))
        hashes[str(parent_path)], hashes[str(path)] = parent['plan_hash'], plan['plan_hash']
    manifest = {'created_at': time.time(), 'project': str(root), 'source_hashes': sources,
        'retired_models': [RETIRED], 'target': TARGET, 'old_campaigns': old_paths, 'new_campaigns': new_paths,
        'plan_hashes': hashes, 'handoff_hash': digest(read(control / 'handoff.json')),
        'smoke': str(smoke), 'smoke_hash': digest(proof), 'unit_hash': digest(tests), 'unit_proof': str(unit_proof),
        'execution_manifest': str(root / 'data/supplement_isolated_execution_v1/manifest.json'),
        'max_attempts_per_new_job': 1, 'inference_api': False,
        'failure_policy': 'Quarantine model on infrastructure error; retain all missing jobs. No accuracy-based retries.',
        'scope': 'New full GPT-OSS A/B/C/D matrix plus unchanged non-Gemma jobs from original plans; original primary retained'}
    manifest['manifest_hash'] = digest(manifest)
    atomic_json(manifest_path, manifest)
    print(json.dumps({'manifest': str(manifest_path), 'gptoss_jobs': sum(len(load_plan(Path(p))['jobs']) for p in new_paths)}), flush=True)


def load(path):
    manifest = read(path)
    wanted = manifest.pop('manifest_hash')
    if digest(manifest) != wanted or manifest['retired_models'] != [RETIRED]:
        raise RuntimeError('Migration or retirement policy changed')
    manifest['manifest_hash'] = wanted
    for raw, checksum in manifest['source_hashes'].items():
        if digest(Path(raw).read_text()) != checksum:
            raise RuntimeError('Migration source changed: ' + raw)
    control = path.parent
    if digest(read(control / 'handoff.json')) != manifest['handoff_hash']:
        raise RuntimeError('Handoff changed')
    if digest(read(manifest['smoke'])) != manifest['smoke_hash'] or digest(read(manifest['unit_proof'])) != manifest['unit_hash']:
        raise RuntimeError('Validation proof changed')
    return manifest


def eligible_jobs(plans, active, blocked):
    ready = []
    for index, (cp, plan) in enumerate(plans):
        gates = {}
        for job in plan['jobs']:
            if job['model'] == RETIRED or job['model'] in blocked or (str(cp), job['id']) in active:
                continue
            folder = cp.parent / 'jobs' / job['id']
            if (folder / 'result.json').exists() or list(folder.glob('attempt[0-9]*')):
                continue
            if job['stage'] == 'test':
                if job['model'] not in gates:
                    gates[job['model']] = model_gate(cp, plan, job['model'])
                if gates[job['model']] != 'passed':
                    continue
            priority = 0 if job['stage'] == 'qualification' else 1 if job['model'] == TARGET else 2
            ready.append((priority, index, cp, plan, job))
    return ready


def adopt(row, plans, manifest, journal, control):
    current = process_cmdline(row['pid'])
    if current:
        if row['out'] not in current:
            raise RuntimeError('Handoff PID reused; inspect before adoption')
        return False
    out = Path(row['out'])
    job = row['config']['job']
    cp, plan = next((cp, p) for cp, p in plans if p['plan_hash'] == row['config']['plan_hash'])
    if read(out / 'config.json') != row['config']:
        raise RuntimeError('Handoff attempt configuration changed')
    with FileLease(cp.parent / 'locks' / (job['id'] + '.lock')):
        destination = out.parent / 'result.json'
        existing = read(destination)
        if existing and existing.get('state') == 'complete':
            verify_completion(Path(existing['run']), plan, job)
            return True
        if (out / 'complete.json').exists():
            proof = verify_completion(out, plan, job)
            result = {'state': 'complete', 'episodes': proof['episodes'], 'audit_hash': digest(proof)}
        else:
            result = {'state': 'quarantined', 'episodes': 0, 'error': read(out / 'failure.json'),
                'reason': 'Preserved child ended without audited completion; no blind retry'}
        result.update(job=job['id'], model=job['model'], run=str(out), worker=row['worker']['name'],
            migration_manifest_hash=manifest['manifest_hash'], adopted_after_dispatcher_retirement=True,
            original_owner=row, prior_result=existing)
        journal.persist(destination, result)
        journal.persist(control / 'adopted' / (job['id'] + '.json'), result)
    return True


def attempt(path, cp, job, worker, journal, inherited):
    manifest = load(path)
    plan = load_plan(cp)
    if job['model'] == RETIRED:
        raise RuntimeError('Gemma is permanently retired in this revision')
    folder = cp.parent / 'jobs' / job['id']
    with FileLease(cp.parent / 'locks' / (job['id'] + '.lock')) as lease:
        if (folder / 'result.json').exists() or list(folder.glob('attempt[0-9]*')):
            raise RuntimeError('Never overwrite/retry prior evidence')
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / 'attempt1'
        if job['model'] == TARGET:
            cmd = [plan['python'], '-u', '-m', 'medtoolbench.supplement.gptoss_runner']
        else:
            cmd = [plan['python'], '-u', '-m', 'medtoolbench.supplement.isolated_runner',
                '--execution-manifest', manifest['execution_manifest']]
        cmd += ['--plan', str(cp), '--job', job['id'], '--worker', worker['name'], '--out', str(out)]
        journal.persist(folder / 'gptoss_migration_command.json', {'command': cmd, 'worker': worker,
            'manifest_hash': manifest['manifest_hash'], 'scientific_plan_hash': plan['plan_hash']})
        outcome = execute(cmd, folder / 'attempt1.gptoss_migration.log', worker, path.parent,
            [*inherited, lease.fileno()], journal)
        result = {'job': job['id'], 'model': job['model'], 'worker': worker['name'], 'run': str(out),
            'attempts': [outcome], 'migration_manifest_hash': manifest['manifest_hash']}
        if outcome['returncode'] == 0 and (out / 'complete.json').exists():
            proof = verify_completion(out, plan, job)
            result.update(state='complete', episodes=proof['episodes'])
        else:
            if (out / 'audit.json').exists() or (out / 'complete.json').exists():
                raise RuntimeError('Audit-invalid result; do not retry or accept')
            result.update(state='quarantined', episodes=0, error=read(out / 'failure.json'),
                assigned_episodes=len(job['episodes']))
        journal.persist(folder / 'result.json', result)
        return result


def finalize(path, manifest, plans):
    """Analysis-only views reference audited source runs, not relabeled baselines."""
    from .report import generate
    for index in range(4):
        old_path, old = plans[index]
        new_path, new = plans[index + 4]
        view = copy.deepcopy(old)
        view.pop('plan_hash')
        view['systems'] = [s for s in old['systems'] if s['name'] != RETIRED] + new['systems']
        view['jobs'] = [j for j in old['jobs'] if j['model'] != RETIRED] + new['jobs']
        view['analysis_only'] = True
        view['gptoss_replacement_manifest'] = str(path)
        view['source_hashes'].update(manifest['source_hashes'])
        view['source_plans'] = {str(old_path): old['plan_hash'], str(new_path): new['plan_hash']}
        view['plan_hash'] = digest(view)
        report_path = path.parent / ('combined_' + 'abcd'[index]) / 'plan.json'
        if report_path.exists():
            raise RuntimeError('Partial combined report exists; inspect without overwriting')
        atomic_json(report_path, view)
        for cp, plan in ((old_path, old), (new_path, new)):
            for job in plan['jobs']:
                if job['model'] == RETIRED:
                    continue
                source = cp.parent / 'jobs' / job['id'] / 'result.json'
                result = read(source)
                verify_completion(Path(result['run']), plan, job)
                atomic_json(report_path.parent / 'jobs' / job['id'] / 'result.json', {**result,
                    'analysis_reference_only': True, 'source_result': str(source),
                    'source_result_hash': digest(result), 'source_plan_hash': plan['plan_hash']})
        generate(report_path)
    atomic_json(path.parent / 'complete.json', {'all_pass': True, 'manifest_hash': manifest['manifest_hash'],
        'original_gemma_matrices_remain_historical': True, 'time': time.time()})


def run(path):
    manifest = load(path)
    control = path.parent
    handoff = read(control / 'handoff.json')
    if process_cmdline(handoff['old_controller_pid']):
        raise RuntimeError('Old dispatcher still exists; do not race it')
    plans = [(Path(p), load_plan(Path(p))) for p in manifest['old_campaigns'] + manifest['new_campaigns']]
    if any(p['plan_hash'] != manifest['plan_hashes'][str(cp)] for cp, p in plans):
        raise RuntimeError('Source plan identity changed')
    workers = plans[0][1]['workers']
    for w in workers:
        validate_worker(plans[0][1], w['name'])
    journal = Journal(control, Path(tempfile.mkdtemp(prefix='medtoolbench-gptoss-')))
    remaining = list(handoff['live_children_preserved'])
    blocked, busy, cursor, fatal = {}, {}, 0, None
    with FileLease(control / 'supervisor.lock') as owner, ThreadPoolExecutor(max_workers=8) as pool:
        if (control / 'started.json').exists():
            raise RuntimeError('A prior migration run needs explicit reconciliation before restart')
        journal.persist(control / 'started.json', {'pid': os.getpid(), 'manifest_hash': manifest['manifest_hash'], 'time': time.time()})
        while True:
            try:
                load(path)
                for row in list(remaining):
                    if adopt(row, plans, manifest, journal, control):
                        remaining.remove(row)
                for name, (future, key) in list(busy.items()):
                    if future.done():
                        # Remove a finished future even if its exception stops dispatch.
                        del busy[name]
                        result = future.result()
                        if result['state'] != 'complete':
                            blocked[result['model']] = result
                for cp, plan in plans:
                    for job in plan['jobs']:
                        if job['model'] == RETIRED:
                            continue
                        result = read(cp.parent / 'jobs' / job['id'] / 'result.json')
                        if result and result.get('state') != 'complete':
                            blocked[job['model']] = result
                if fatal is None and journal.healthy():
                    reserved = {r['worker']['name'] for r in remaining} | set(busy)
                    active = {key for _, key in busy.values()}
                    for worker in workers:
                        if worker['name'] in reserved:
                            continue
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(worker['gpu'])
                        try:
                            with AllocatedGpuLease(worker):
                                pass
                        except BlockingIOError:
                            continue
                        ready = eligible_jobs(plans, active, blocked)
                        if not ready:
                            continue
                        ready.sort(key=lambda r: (r[0], (r[1] - cursor) % len(plans), r[4]['id']))
                        _, index, cp, plan, job = ready[0]
                        cursor = (index + 1) % len(plans)
                        key = (str(cp), job['id'])
                        busy[worker['name']] = (pool.submit(attempt, path, cp, job, worker, journal, [owner.fileno()]), key)
                        active.add(key)
                pending, completed = [], 0
                for cp, plan in plans:
                    for job in plan['jobs']:
                        if job['model'] == RETIRED:
                            continue
                        result = read(cp.parent / 'jobs' / job['id'] / 'result.json')
                        if result and result.get('state') == 'complete':
                            completed += len(job['episodes'])
                        else:
                            pending.append({'plan': str(cp), 'job': job['id'], 'model': job['model'],
                                'stage': job['stage'], 'episodes': len(job['episodes'])})
                journal.put(control / 'pending.json', {'jobs': pending, 'completed_including_development': completed,
                    'retired_model_excluded_explicitly': RETIRED, 'time': time.time()})
                journal.put(control / 'status.json', {'state': 'integrity_stop' if fatal else 'running',
                    'pid': os.getpid(), 'busy': {n: list(k) for n, (_, k) in busy.items()},
                    'preserved_children': [r['pid'] for r in remaining], 'blocked_models': list(blocked),
                    'retired_models': [RETIRED], 'heartbeat': time.time(), 'manifest_hash': manifest['manifest_hash']})
                if not busy and not remaining:
                    if fatal or pending:
                        journal.persist(control / 'needs_review.json', {'error': fatal, 'blocked_models': blocked,
                            'pending_jobs': len(pending), 'time': time.time()})
                        return
                    break
            except Exception:
                fatal = traceback.format_exc()
                journal.put(control / 'integrity_failure.json', {'error': fatal, 'time': time.time()})
                if not busy and not remaining:
                    journal.persist(control / 'needs_review.json', {'error': fatal, 'time': time.time()})
                    return
            time.sleep(10)
        finalize(path, manifest, plans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--control', type=Path, default=Path('data/gptoss_replacement_v1'))
    parser.add_argument('--smoke', type=Path, default=Path('data/gptoss_replacement_v1/smoke_v1/complete.json'))
    parser.add_argument('--unit-proof', type=Path, default=Path('data/gptoss_replacement_v1/unit_tests_release.json'))
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.control.resolve(), args.smoke.resolve(), args.unit_proof.resolve())
    else:
        run(args.control.resolve() / 'manifest.json')
