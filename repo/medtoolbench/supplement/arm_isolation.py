"""Operational continuation with exact-evidence, arm-local quarantine.

Original plans, failed attempts, inference and graders stay immutable. Only the
reviewed concurrent-writer failures cease blocking unrelated arms. Any other
failure still quarantines its model. Deferred jobs remain in the denominator.
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
from . import gptoss_pipeline as engine
from .non_gptoss_priority import ALLOWED, DEFERRED, load as load_prior, require_allowed
from .runner import load_plan
from .storage_recovery import Journal, atomic_json, read

ROOT = Path(__file__).resolve().parents[3]
_ATTEMPT = engine.attempt


def isolated(job):
    return bool(job.get('settings', {}).get('stress', {}).get('concurrent'))


def source_hashes():
    files = [Path(__file__).resolve(), ROOT / 'repo/tests/test_arm_isolation.py',
             ROOT / 'repo/scripts/validate_arm_isolation.py']
    return {str(p): digest(p.read_text()) for p in files}


def handoff(control, prior):
    """Stop only the verified supervisor, preserving every active model child."""
    destination = control / 'handoff.json'
    if destination.exists():
        raise RuntimeError('Never overwrite a handoff')
    load_prior(prior / 'manifest.json')
    status = read(prior / 'status.json')
    pid = status['pid']
    command = engine.process_cmdline(pid)
    if 'repo/scripts/run_non_gptoss_priority.py ' not in command:
        raise RuntimeError('Unexpected predecessor controller identity')
    os.kill(pid, signal.SIGSTOP)
    try:
        status = read(prior / 'status.json')
        rows = []
        for name, (cp, job_id) in status['busy'].items():
            row = read(prior / 'workers' / (name + '.json'))
            config = read(Path(row['out']) / 'config.json')
            if config['job']['id'] != job_id or config['plan_hash'] != load_plan(Path(cp))['plan_hash']:
                raise RuntimeError('Busy child snapshot mismatch')
            current = engine.process_cmdline(row['pid'])
            if current and (row['out'] not in current or '-m medtoolbench.supplement.isolated_runner ' not in current):
                raise RuntimeError('Child identity changed')
            require_allowed(config['job']['model'])
            rows.append({**row, 'config': config, 'observed_cmdline': current})
        for row in read(prior / 'handoff.json')['live_children_preserved']:
            if row['pid'] in status.get('preserved_children', []):
                rows.append(row)
        atomic_json(destination, {'time': time.time(), 'old_controller_pid': pid,
            'old_controller_cmdline': command, 'old_status': status, 'old_control': str(prior),
            'live_children_preserved': rows, 'no_model_children_signalled': True})
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
        raise RuntimeError('Predecessor did not exit; do not start successor')
    atomic_json(prior / 'superseded_arm_isolation.json', {'successor': str(control),
        'handoff': str(destination), 'time': time.time(), 'preserve_all_evidence': True})


def prepare(control, unit):
    if (control / 'manifest.json').exists():
        raise RuntimeError('Never overwrite a continuation manifest')
    proof = read(unit)
    if not proof or not proof.get('all_pass') or proof.get('source_hashes') != source_hashes():
        raise RuntimeError('Passing exact-source CPU regression required')
    prior_path = ROOT / 'data/non_gptoss_priority_v1/manifest.json'
    prior = load_prior(prior_path)
    snapshot = read(control / 'handoff.json')
    if not snapshot or engine.process_cmdline(snapshot['old_controller_pid']):
        raise RuntimeError('Verified dispatcher handoff required')
    reviewed, deferred = {}, []
    for raw in prior['old_campaigns']:
        cp, plan = Path(raw), load_plan(Path(raw))
        for job in plan['jobs']:
            if job['model'] not in ALLOWED:
                continue
            file = cp.parent / 'jobs' / job['id'] / 'result.json'
            result = read(file)
            if isolated(job) and (not result or result.get('state') != 'complete'):
                deferred.append({'plan': raw, 'job': job['id'], 'model': job['model'],
                                 'episodes': len(job['episodes']), 'stage': job['stage']})
            if result and result.get('state') != 'complete':
                error = (result.get('error') or {}).get('error', '')
                if not isolated(job) or not error.endswith('RuntimeError: Controlled concurrent write failed\n'):
                    raise RuntimeError('Unreviewed predecessor failure: ' + str(file))
                reviewed[str(file)] = digest(result)
    if len(reviewed) != 2:
        raise RuntimeError('Expected the two individually reviewed incident records')
    manifest = {k: prior[k] for k in ('project', 'old_campaigns', 'new_campaigns', 'plan_hashes',
        'execution_manifest', 'allowed_models', 'deferred_models', 'retired_models')}
    manifest.update(created_at=time.time(), source_hashes=source_hashes(),
        prior_manifest=str(prior_path), prior_manifest_hash=prior['manifest_hash'],
        handoff_hash=digest(snapshot), unit=str(unit), unit_hash=digest(proof),
        reviewed_failures=reviewed, deferred_jobs=deferred, original_denominator_preserved=True,
        automatic_deferred_resume=False, inference_api=False,
        failure_policy='Only exact reviewed concurrent failures are arm-local; unknown failures still block their model')
    manifest['manifest_hash'] = digest(manifest)
    atomic_json(control / 'manifest.json', manifest)


def load(path):
    m = read(path)
    wanted = m.pop('manifest_hash')
    if digest(m) != wanted:
        raise RuntimeError('Arm-isolation manifest changed')
    m['manifest_hash'] = wanted
    prior = load_prior(Path(m['prior_manifest']))
    if prior['manifest_hash'] != m['prior_manifest_hash']:
        raise RuntimeError('Predecessor changed')
    for file, checksum in m['source_hashes'].items():
        if digest(Path(file).read_text()) != checksum:
            raise RuntimeError('Arm-isolation source changed: ' + file)
    for file, checksum in m['reviewed_failures'].items():
        if digest(read(file)) != checksum:
            raise RuntimeError('Quarantined evidence changed')
    if digest(read(path.parent / 'handoff.json')) != m['handoff_hash'] or digest(read(m['unit'])) != m['unit_hash']:
        raise RuntimeError('Handoff or validation changed')
    return m


def blockers(plans, reviewed):
    blocked = {}
    for cp, plan in plans:
        for job in plan['jobs']:
            if job['model'] not in ALLOWED:
                continue
            file = cp.parent / 'jobs' / job['id'] / 'result.json'
            result = read(file)
            if result and result.get('state') != 'complete':
                if isolated(job) and reviewed.get(str(file)) == digest(result):
                    continue
                blocked[job['model']] = result
    return blocked


def eligible(plans, active, blocked):
    return [row for row in engine.eligible_jobs(plans, active, blocked)
            if row[4]['model'] in ALLOWED and not isolated(row[4])]


def attempt(path, cp, job, worker, journal, inherited):
    require_allowed(job['model'])
    if isolated(job):
        raise RuntimeError('Concurrent-writer arm requires a separately validated repair')
    return _ATTEMPT(path, cp, job, worker, journal, inherited)


def run(path):
    manifest = load(path)
    engine.load = load  # Exact same worker launcher and independent audit.
    control = path.parent
    snapshot = read(control / 'handoff.json')
    if engine.process_cmdline(snapshot['old_controller_pid']):
        raise RuntimeError('Predecessor still exists')
    plans = [(Path(p), load_plan(Path(p))) for p in manifest['old_campaigns']]
    if any(plan['plan_hash'] != manifest['plan_hashes'][str(cp)] for cp, plan in plans):
        raise RuntimeError('Scientific plans changed')
    workers = plans[0][1]['workers']
    for worker in workers:
        validate_worker(plans[0][1], worker['name'])
    journal = Journal(control, Path(tempfile.mkdtemp(prefix='medtoolbench-arm-isolation-')))
    remaining, busy, fatal, cursor = list(snapshot['live_children_preserved']), {}, None, 0
    with FileLease(control / 'supervisor.lock') as owner, ThreadPoolExecutor(max_workers=8) as pool:
        if (control / 'started.json').exists():
            raise RuntimeError('Explicit reconciliation required before restarting')
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
                blocked = blockers(plans, manifest['reviewed_failures'])
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
                        ready = eligible(plans, active, blocked)
                        if not ready:
                            continue
                        ready.sort(key=lambda r: (r[0], (r[1] - cursor) % len(plans), r[4]['id']))
                        _, index, cp, plan, job = ready[0]
                        cursor = (index + 1) % len(plans)
                        key = (str(cp), job['id'])
                        busy[worker['name']] = (pool.submit(attempt, path, cp, job, worker,
                            journal, [owner.fileno()]), key)
                        active.add(key)
                pending, complete = [], 0
                for cp, plan in plans:
                    for job in plan['jobs']:
                        if job['model'] not in ALLOWED:
                            continue
                        result = read(cp.parent / 'jobs' / job['id'] / 'result.json')
                        if result and result.get('state') == 'complete':
                            complete += len(job['episodes'])
                        else:
                            pending.append({'plan': str(cp), 'job': job['id'], 'model': job['model'],
                                'episodes': len(job['episodes']), 'stage': job['stage'], 'isolated': isolated(job)})
                journal.put(control / 'pending.json', {'jobs': pending,
                    'completed_including_development': complete, 'time': time.time()})
                journal.put(control / 'status.json', {'state': 'integrity_stop' if fatal else 'running',
                    'pid': os.getpid(), 'busy': {n: list(k) for n, (_, k) in busy.items()},
                    'preserved_children': [r['pid'] for r in remaining], 'blocked_models': list(blocked),
                    'isolated_arm': 'concurrent_writer', 'deferred_episodes': sum(r['episodes'] for r in pending if r['isolated']),
                    'heartbeat': time.time(), 'manifest_hash': manifest['manifest_hash']})
                if not busy and not remaining:
                    journal.persist(control / 'needs_review.json', {'error': fatal, 'blocked_models': blocked,
                        'pending_jobs': len(pending), 'entire_project_complete': not pending, 'time': time.time()})
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
    p.add_argument('--control', type=Path, default=ROOT / 'data/non_gptoss_priority_v2')
    p.add_argument('--unit', type=Path)
    a = p.parse_args()
    if a.command == 'handoff':
        handoff(a.control.resolve(), ROOT / 'data/non_gptoss_priority_v1')
    elif a.command == 'prepare':
        prepare(a.control.resolve(), a.unit.resolve())
    else:
        run(a.control.resolve() / 'manifest.json')
