"""Resume certified concurrent jobs alongside unchanged healthy experiments.

Repaired results are an explicit overlay: original failed records and partial
rows are never overwritten or pooled. The full original denominator is kept.
"""
import argparse
import os
import signal
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import AllocatedGpuLease, FileLease, validate_worker
from . import arm_isolation as isolation, gptoss_pipeline as engine
from .isolated_execution import model_gate
from .memory_execution import verify_completion
from .non_gptoss_priority import ALLOWED, require_allowed
from .runner import load_plan
from .storage_recovery import Journal, atomic_json, execute, read

ROOT = Path(__file__).resolve().parents[3]
_ORIGINAL_ATTEMPT = engine.attempt


def repair_key(cp, job_id):
    return str(Path(cp).resolve()) + '::' + job_id


def source_hashes():
    files = [Path(__file__).resolve(), ROOT / 'repo/tests/test_concurrent_pipeline.py',
             ROOT / 'repo/scripts/validate_concurrent_pipeline.py']
    return {str(p): digest(p.read_text()) for p in files}


def check_validation(proof):
    if (not proof or not proof.get('all_pass') or proof.get('held_out_tasks_used') is not False
            or proof.get('model_inference') is not False or not proof.get('clinical_accuracy_not_a_gate')):
        raise RuntimeError('Synthetic transport validation required')
    checks = proof.get('checks', [])
    expected = {(p, h, b) for p in ('vanilla', 'informed', 'evidence_only', 'medguard', 'no_read', 'no_write', 'no_retry')
                for h in ('commit_timeout', 'no_commit_timeout') for b in ('encoded_date', 'resource_mismatch')}
    if len(checks) != 28 or {(r['policy'], r['hazard'], r['malformed']) for r in checks if r['all_pass']} != expected:
        raise RuntimeError('Incomplete malformed-request/audit coverage')
    equivalence = proof.get('positive_path_equivalence', [])
    if len(equivalence) != 14 or {(r['policy'], r['hazard']) for r in equivalence if r['all_pass']} != {(p, h) for p, h, _ in expected}:
        raise RuntimeError('Incomplete positive-path equivalence')
    if set(proof.get('old_crashes_reproduced', [])) != {'encoded_date', 'resource_mismatch'}:
        raise RuntimeError('Missing frozen failure reproduction')
    for file, checksum in proof['source_hashes'].items():
        if digest(Path(file).read_text()) != checksum:
            raise RuntimeError('Transport validation source changed')


def handoff(control):
    prior = ROOT / 'data/non_gptoss_priority_v2'
    isolation.load(prior / 'manifest.json')
    if (control / 'handoff.json').exists():
        raise RuntimeError('Never overwrite ownership evidence')
    status = read(prior / 'status.json')
    pid = status['pid']
    command = engine.process_cmdline(pid)
    if '-m medtoolbench.supplement.arm_isolation run' not in command:
        raise RuntimeError('Unexpected prior dispatcher identity')
    os.kill(pid, signal.SIGSTOP)
    try:
        status = read(prior / 'status.json')
        preserved = []
        for name, (cp, job_id) in status['busy'].items():
            row = read(prior / 'workers' / (name + '.json'))
            config = read(Path(row['out']) / 'config.json')
            if config['job']['id'] != job_id or config['plan_hash'] != load_plan(Path(cp))['plan_hash']:
                raise RuntimeError('Child launch still changing; resume and retry handoff later')
            current = engine.process_cmdline(row['pid'])
            if current and (row['out'] not in current or '-m medtoolbench.supplement.isolated_runner ' not in current):
                raise RuntimeError('Unexpected child identity')
            require_allowed(config['job']['model'])
            preserved.append({**row, 'config': config, 'observed_cmdline': current})
        for row in read(prior / 'handoff.json')['live_children_preserved']:
            if row['pid'] in status['preserved_children']:
                current = engine.process_cmdline(row['pid'])
                if current and row['out'] not in current:
                    raise RuntimeError('Adopted child PID reused')
                preserved.append(row)
        atomic_json(control / 'handoff.json', {'time': time.time(), 'old_controller_pid': pid,
            'old_controller_cmdline': command, 'old_status': status,
            'live_children_preserved': preserved, 'no_model_children_signalled': True})
    except BaseException:
        os.kill(pid, signal.SIGCONT)
        raise
    os.kill(pid, signal.SIGTERM)
    os.kill(pid, signal.SIGCONT)
    for _ in range(50):
        if not engine.process_cmdline(pid):
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Old dispatcher did not exit')
    atomic_json(prior / 'superseded_concurrent_repair.json', {'successor': str(control),
        'time': time.time(), 'preserve_all_evidence': True})


def prepare(control, unit, validation):
    if (control / 'manifest.json').exists():
        raise RuntimeError('Never overwrite repair manifest')
    proof, transport = read(unit), read(validation)
    if not proof or not proof.get('all_pass') or proof.get('source_hashes') != source_hashes():
        raise RuntimeError('Exact-source full scheduling regression required')
    check_validation(transport)
    prior_path = ROOT / 'data/non_gptoss_priority_v2/manifest.json'
    prior = isolation.load(prior_path)
    snapshot = read(control / 'handoff.json')
    if not snapshot or engine.process_cmdline(snapshot['old_controller_pid']):
        raise RuntimeError('Verified ownership handoff required')
    jobs = {}
    for raw in prior['old_campaigns']:
        cp, plan = Path(raw), load_plan(Path(raw))
        for job in plan['jobs']:
            if job['model'] not in ALLOWED or not isolation.isolated(job):
                continue
            folder = cp.parent / 'jobs' / job['id']
            result = read(folder / 'result.json')
            if result and result.get('state') == 'complete':
                verify_completion(Path(result['run']), plan, job)
                continue
            if result:
                if prior['reviewed_failures'].get(str(folder / 'result.json')) != digest(result):
                    raise RuntimeError('Unreviewed failed job cannot be retried')
                if any((p / marker).exists() for p in folder.glob('attempt[0-9]*') for marker in ('audit.json', 'complete.json')):
                    raise RuntimeError('Never retry audited scientific evidence')
            elif list(folder.glob('attempt[0-9]*')):
                raise RuntimeError('Unresolved attempt is not repairable')
            jobs[repair_key(cp, job['id'])] = {'plan': str(cp), 'job': job['id'], 'model': job['model'],
                'episodes': len(job['episodes']), 'original_result': str(folder / 'result.json'),
                'original_result_hash': digest(result) if result else None,
                'partial_rows_pooled': False, 'new_attempts_allowed': 1}
    if {r['job'] for r in jobs.values()} != {r['job'] for r in prior['deferred_jobs']}:
        raise RuntimeError('Repair selection must cover all deferred jobs, not selected outcomes')
    manifest = {k: prior[k] for k in ('project', 'old_campaigns', 'new_campaigns', 'plan_hashes',
        'execution_manifest', 'allowed_models', 'deferred_models', 'retired_models')}
    manifest.update(created_at=time.time(), source_hashes=source_hashes(),
        repair_source_hashes=transport['source_hashes'], transport_validation=str(validation),
        transport_validation_hash=digest(transport), prior_manifest=str(prior_path),
        prior_manifest_hash=prior['manifest_hash'], handoff_hash=digest(snapshot),
        unit=str(unit), unit_hash=digest(proof), repair_jobs=jobs,
        original_denominator_preserved=True, original_failed_records_immutable=True,
        automatic_deferred_resume=False, inference_api=False, max_new_attempts_per_job=1,
        failure_policy='No blind retry; all new failures quarantine the affected model',
        repair_selection='All unresolved concurrent-writer jobs; no accuracy or final-answer selection')
    manifest['manifest_hash'] = digest(manifest)
    atomic_json(control / 'manifest.json', manifest)


def load(path):
    m = read(path)
    wanted = m.pop('manifest_hash')
    if digest(m) != wanted:
        raise RuntimeError('Concurrent continuation manifest changed')
    m['manifest_hash'] = wanted
    if isolation.load(Path(m['prior_manifest']))['manifest_hash'] != m['prior_manifest_hash']:
        raise RuntimeError('Prior continuation changed')
    for file, checksum in {**m['source_hashes'], **m['repair_source_hashes']}.items():
        if digest(Path(file).read_text()) != checksum:
            raise RuntimeError('Certified source changed: ' + file)
    if (digest(read(path.parent / 'handoff.json')) != m['handoff_hash']
            or digest(read(m['unit'])) != m['unit_hash']
            or digest(read(m['transport_validation'])) != m['transport_validation_hash']):
        raise RuntimeError('Ownership or validation evidence changed')
    for row in m['repair_jobs'].values():
        original = read(row['original_result'])
        if (digest(original) if original else None) != row['original_result_hash']:
            raise RuntimeError('Original failed/pending result was modified')
    return m


def folder_for(control, cp, job, manifest):
    if repair_key(cp, job['id']) in manifest['repair_jobs']:
        return control / 'repair_jobs' / job['id']
    return cp.parent / 'jobs' / job['id']


def inventory(control, plans, manifest):
    pending, blocked, completed = [], {}, 0
    for cp, plan in plans:
        for job in plan['jobs']:
            if job['model'] not in ALLOWED:
                continue
            folder = folder_for(control, cp, job, manifest)
            result = read(folder / 'result.json')
            if result and result.get('state') == 'complete':
                completed += len(job['episodes'])
            else:
                pending.append({'plan': str(cp), 'job': job['id'], 'model': job['model'],
                    'stage': job['stage'], 'episodes': len(job['episodes']),
                    'repair': repair_key(cp, job['id']) in manifest['repair_jobs']})
                if result:
                    blocked[job['model']] = result
    return pending, blocked, completed


def eligible(control, plans, manifest, active, blocked):
    ready = []
    for index, (cp, plan) in enumerate(plans):
        gates = {}
        for job in plan['jobs']:
            if job['model'] not in ALLOWED or job['model'] in blocked or (str(cp), job['id']) in active:
                continue
            repair = repair_key(cp, job['id']) in manifest['repair_jobs']
            if isolation.isolated(job) and not repair:
                continue  # Completed pre-repair concurrent evidence is never rerun.
            folder = folder_for(control, cp, job, manifest)
            if (folder / 'result.json').exists() or list(folder.glob('attempt[0-9]*')):
                continue
            if job['stage'] == 'test':
                if job['model'] not in gates:
                    gates[job['model']] = model_gate(cp, plan, job['model'])
                if gates[job['model']] != 'passed':
                    continue
            ready.append((0 if job['stage'] == 'qualification' else 1, 0 if repair else 1, index, cp, plan, job))
    return ready


def attempt(path, cp, job, worker, journal, inherited):
    require_allowed(job['model'])
    manifest = load(path)
    key = repair_key(cp, job['id'])
    if key not in manifest['repair_jobs']:
        if isolation.isolated(job):
            raise RuntimeError('Concurrent job was not authorized')
        return _ORIGINAL_ATTEMPT(path, cp, job, worker, journal, inherited)
    plan = load_plan(cp)
    folder = folder_for(path.parent, cp, job, manifest)
    with FileLease(cp.parent / 'locks' / (job['id'] + '.lock')) as lease:
        if (folder / 'result.json').exists() or list(folder.glob('attempt[0-9]*')):
            raise RuntimeError('No blind repair retries or overwrite')
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / 'attempt1'
        cmd = [plan['python'], '-u', '-m', 'medtoolbench.supplement.concurrent_runner',
            '--repair-manifest', str(path), '--plan', str(cp), '--job', job['id'],
            '--worker', worker['name'], '--out', str(out)]
        journal.persist(folder / 'command.json', {'command': cmd, 'worker': worker,
            'manifest_hash': manifest['manifest_hash'], 'authorization': manifest['repair_jobs'][key]})
        outcome = execute(cmd, folder / 'attempt1.log', worker, path.parent,
                          [*inherited, lease.fileno()], journal)
        result = {'job': job['id'], 'model': job['model'], 'worker': worker['name'],
            'run': str(out), 'attempts': [outcome], 'manifest_hash': manifest['manifest_hash'],
            'original_plan': str(cp), 'original_evidence': manifest['repair_jobs'][key],
            'prior_partial_rows_pooled': False}
        if outcome['returncode'] == 0 and (out / 'complete.json').exists():
            proof = verify_completion(out, plan, job)
            if read(out / 'config.json').get('concurrent_repair', {}).get('manifest_hash') != manifest['manifest_hash']:
                raise RuntimeError('Missing exact repair execution identity')
            result.update(state='complete', episodes=proof['episodes'])
        else:
            if (out / 'audit.json').exists() or (out / 'complete.json').exists():
                raise RuntimeError('Audit-invalid repair is not accepted or retried')
            result.update(state='quarantined', episodes=0, error=read(out / 'failure.json'),
                          assigned_episodes=len(job['episodes']))
        journal.persist(folder / 'result.json', result)
        return result


def finalize(control, plans, manifest):
    rows = []
    for cp, plan in plans:
        for job in plan['jobs']:
            if job['model'] not in ALLOWED:
                continue
            file = folder_for(control, cp, job, manifest) / 'result.json'
            result = read(file)
            if not result or result.get('state') != 'complete':
                raise RuntimeError('Unresolved jobs remain in full denominator')
            verify_completion(Path(result['run']), plan, job)
            rows.append({'plan': str(cp), 'job': job['id'], 'stage': job['stage'],
                'model': job['model'], 'episodes': len(job['episodes']), 'result': str(file),
                'result_hash': digest(result), 'repaired': repair_key(cp, job['id']) in manifest['repair_jobs']})
    atomic_json(control / 'complete.json', {'all_pass': True, 'time': time.time(),
        'manifest_hash': manifest['manifest_hash'], 'jobs': rows,
        'formal_episodes': sum(r['episodes'] for r in rows if r['stage'] == 'test'),
        'development_episodes': sum(r['episodes'] for r in rows if r['stage'] == 'qualification'),
        'scope': 'Other-model supplemental cohort only', 'entire_project_complete': False,
        'automatic_deferred_resume': False, 'deferred_models': manifest['deferred_models']})


def run(path):
    manifest = load(path)
    engine.load = load
    control = path.parent
    snapshot = read(control / 'handoff.json')
    if engine.process_cmdline(snapshot['old_controller_pid']):
        raise RuntimeError('Prior dispatcher still exists')
    plans = [(Path(p), load_plan(Path(p))) for p in manifest['old_campaigns']]
    if any(p['plan_hash'] != manifest['plan_hashes'][str(cp)] for cp, p in plans):
        raise RuntimeError('Original scientific plan changed')
    workers = plans[0][1]['workers']
    for worker in workers:
        validate_worker(plans[0][1], worker['name'])
    journal = Journal(control, Path(tempfile.mkdtemp(prefix='medtoolbench-concurrent-repair-')))
    remaining, busy, fatal, cursor = list(snapshot['live_children_preserved']), {}, None, 0
    with FileLease(control / 'supervisor.lock') as owner, ThreadPoolExecutor(max_workers=8) as pool:
        if (control / 'started.json').exists():
            raise RuntimeError('Restart requires explicit ownership reconciliation')
        journal.persist(control / 'started.json', {'pid': os.getpid(), 'time': time.time()})
        while True:
            try:
                load(path)
                for row in list(remaining):
                    if engine.adopt(row, plans, manifest, journal, control):
                        remaining.remove(row)
                for name, (future, key) in list(busy.items()):
                    if future.done():
                        del busy[name]
                        future.result()
                pending, blocked, completed = inventory(control, plans, manifest)
                if fatal is None and journal.healthy():
                    reserved = set(busy) | {r['worker']['name'] for r in remaining}
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
                        ready = eligible(control, plans, manifest, active, blocked)
                        if not ready:
                            continue
                        ready.sort(key=lambda r: (r[0], r[1], (r[2] - cursor) % len(plans), r[5]['id']))
                        _, _, index, cp, plan, job = ready[0]
                        cursor = (index + 1) % len(plans)
                        key = (str(cp), job['id'])
                        busy[worker['name']] = (pool.submit(attempt, path, cp, job, worker,
                            journal, [owner.fileno()]), key)
                        active.add(key)
                journal.put(control / 'pending.json', {'jobs': pending,
                    'completed_including_development': completed, 'time': time.time()})
                journal.put(control / 'status.json', {'state': 'integrity_stop' if fatal else 'running',
                    'pid': os.getpid(), 'busy': {n: list(k) for n, (_, k) in busy.items()},
                    'preserved_children': [r['pid'] for r in remaining], 'blocked_models': list(blocked),
                    'repair_pending_episodes': sum(r['episodes'] for r in pending if r['repair']),
                    'original_denominator_preserved': True, 'heartbeat': time.time(),
                    'manifest_hash': manifest['manifest_hash']})
                if not busy and not remaining:
                    if fatal or pending:
                        journal.persist(control / 'needs_review.json', {'error': fatal, 'blocked_models': blocked,
                            'pending_jobs': len(pending), 'time': time.time()})
                    else:
                        finalize(control, plans, manifest)
                    return
            except Exception:
                fatal = traceback.format_exc()
                journal.put(control / 'integrity_failure.json', {'error': fatal, 'time': time.time()})
                if not busy and not remaining:
                    return
            time.sleep(10)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['handoff', 'prepare', 'run'])
    p.add_argument('--control', type=Path, default=ROOT / 'data/non_gptoss_priority_v3')
    p.add_argument('--unit', type=Path)
    p.add_argument('--validation', type=Path, default=ROOT / 'data/concurrent_repair_validation_v2/complete.json')
    a = p.parse_args()
    if a.command == 'handoff':
        handoff(a.control.resolve())
    elif a.command == 'prepare':
        prepare(a.control.resolve(), a.unit.resolve(), a.validation.resolve())
    else:
        run(a.control.resolve() / 'manifest.json')
