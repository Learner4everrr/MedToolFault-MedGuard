"""Qwen-only B continuation: unchanged scientific plan and isolated runner.

Scheduling-only amendment. Preserve certified work, never retry unresolved work,
and stop dispatch after any failure. Other models cannot enter this queue.
"""
import argparse
import os
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import AllocatedGpuLease, FileLease, validate_worker
from .isolated_execution import model_gate
from .isolated_manifest import load as load_execution, read, require_cpu_validation
from .memory_execution import verify_completion
from .runner import load_plan
from .storage_recovery import Journal, atomic_json, execute, write_probe

ROOT = Path(__file__).resolve().parents[3]
CONTROL = ROOT / 'data/b_qwen_priority_v1'
PLAN = ROOT / 'data/supplement_b_v2/plan.json'
EXECUTION = ROOT / 'data/supplement_isolated_execution_v1/manifest.json'
MODEL = 'qwen35_9b'


def source_hashes():
    files = [Path(__file__), ROOT / 'repo/tests/test_b_qwen_priority.py',
             ROOT / 'repo/scripts/validate_b_qwen_priority.py',
             Path(__file__).with_name('storage_recovery.py'),
             Path(__file__).with_name('isolated_execution.py'),
             Path(__file__).with_name('memory_execution.py')]
    return {str(f): digest(f.read_text()) for f in files}


def selected(plan, stage=None):
    return [j for j in plan['jobs'] if j['model'] == MODEL
            and j['arm'] == 'external_qa' and (stage is None or j['stage'] == stage)]


def require_fresh(folder):
    # Any prior evidence requires explicit reconciliation, never a silent rerun.
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError('Job already has evidence: ' + str(folder))


def inventory(cp, plan):
    retained, pending = [], []
    for job in selected(plan):
        folder = cp.parent / 'jobs' / job['id']
        result = read(folder / 'result.json')
        if result and result.get('state') == 'complete':
            proof = verify_completion(Path(result['run']), plan, job)
            retained.append({'job': job['id'], 'stage': job['stage'],
                'episodes': proof['episodes'], 'result_hash': digest(result),
                'audit_hash': digest(proof)})
        else:
            require_fresh(folder)
            if job['stage'] != 'test':
                raise RuntimeError('Qwen development must already be certified')
            pending.append(job['id'])
    return retained, pending


def prepare(control, unit):
    if (control / 'manifest.json').exists():
        raise RuntimeError('Never overwrite a scheduling amendment')
    validation = read(unit)
    if not validation or not validation.get('all_pass') or validation.get('skipped') or validation.get('source_hashes') != source_hashes():
        raise RuntimeError('Exact-source CPU validation required')
    plan = load_plan(PLAN)
    execution = load_execution(EXECUTION)
    require_cpu_validation(EXECUTION.parent, execution)
    if str(PLAN) not in execution['campaigns'] or model_gate(PLAN, plan, MODEL) != 'passed':
        raise RuntimeError('Original B execution/development gate not passed')
    if plan['runner_module'] != 'medtoolbench.supplement.external_runner' or plan['inference_api']:
        raise RuntimeError('Wrong B execution configuration')
    retained, pending = inventory(PLAN, plan)
    atomic_json(control / 'storage_probe.json', write_probe(control / 'storage_probes'))
    manifest = {'created_at': time.time(), 'model': MODEL, 'plan': str(PLAN),
        'plan_hash': plan['plan_hash'], 'execution_manifest': str(EXECUTION),
        'execution_manifest_hash': execution['manifest_hash'],
        'source_hashes': source_hashes(), 'unit': str(unit), 'unit_hash': digest(validation),
        'workers': plan['workers'], 'retained': retained, 'pending': pending,
        'formal_episodes': sum(len(j['episodes']) for j in selected(plan, 'test')),
        'session': 'medtoolbench-b-qwen-v1', 'inference_configuration_unchanged': True,
        'inference_api': False, 'automatic_retries': 0, 'next_model_auto_start': False,
        'policy': 'Qwen B test jobs only, original job order, eight independent GPU lanes; preserve audited original work; drain and stop on any failure.'}
    manifest['manifest_hash'] = digest(manifest)
    atomic_json(control / 'manifest.json', manifest)
    return manifest


def load(path):
    manifest = read(path)
    checksum = manifest.pop('manifest_hash')
    if digest(manifest) != checksum or manifest['source_hashes'] != source_hashes():
        raise RuntimeError('Qwen B scheduler identity changed')
    manifest['manifest_hash'] = checksum
    if manifest['model'] != MODEL or Path(manifest['plan']) != PLAN:
        raise RuntimeError('Non-Qwen or non-B dispatch forbidden')
    if digest(read(manifest['unit'])) != manifest['unit_hash']:
        raise RuntimeError('CPU validation changed')
    plan = load_plan(PLAN)
    execution = load_execution(EXECUTION)
    require_cpu_validation(EXECUTION.parent, execution)
    if plan['plan_hash'] != manifest['plan_hash'] or execution['manifest_hash'] != manifest['execution_manifest_hash'] or manifest['workers'] != plan['workers']:
        raise RuntimeError('Original inference plan changed')
    return manifest, plan


def authorize(manifest, plan, job):
    if job not in selected(plan, 'test') or job['id'] not in manifest['pending']:
        raise RuntimeError('Job outside Qwen-only continuation')
    if model_gate(PLAN, plan, MODEL) != 'passed':
        raise RuntimeError('Original Qwen B gate not passed')
    require_fresh(PLAN.parent / 'jobs' / job['id'])


def attempt(path, job, worker, fds, journal):
    manifest, plan = load(path)
    control = path.parent
    folder = PLAN.parent / 'jobs' / job['id']
    with FileLease(PLAN.parent / 'locks' / (job['id'] + '.lock')) as lease:
        authorize(manifest, plan, job)
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / 'attempt1'
        cmd = [plan['python'], '-u', '-m', 'medtoolbench.supplement.isolated_runner',
            '--execution-manifest', str(EXECUTION), '--plan', str(PLAN),
            '--worker', worker['name'], '--job', job['id'], '--out', str(out)]
        atomic_json(folder / 'b_qwen_authorization.json', {
            'manifest_hash': manifest['manifest_hash'], 'command': cmd,
            'worker': worker, 'time': time.time(), 'attempts_per_job': 1})
        execution = execute(cmd, folder / 'attempt1.qwen_b.log', worker,
            control, [*fds, lease.fileno()], journal)
        result = {'job': job['id'], 'model': MODEL, 'run': str(out),
            'worker': worker['name'], 'attempts': [{'attempt': 1, **execution}],
            'execution_manifest': str(EXECUTION),
            'execution_manifest_hash': manifest['execution_manifest_hash'],
            'scheduling_manifest_hash': manifest['manifest_hash'], 'time': time.time()}
        if execution['returncode'] == 0 and (out / 'complete.json').exists():
            proof = verify_completion(out, plan, job)
            result.update(state='complete', episodes=proof['episodes'], audit_hash=digest(proof))
        else:
            result.update(state='failed', failure=read(out / 'failure.json'))
        journal.persist(folder / 'result.json', result)
        journal.put(control / 'workers' / (worker['name'] + '.json'), {
            'state': 'idle', 'worker': worker, 'finished_job': job['id'], 'heartbeat': time.time()})
        return result


def snapshot(manifest, plan, active, failures, started):
    complete, partial, rows = 0, 0, []
    audited_jobs = set()
    for job in selected(plan, 'test'):
        folder = PLAN.parent / 'jobs' / job['id']
        result = read(folder / 'result.json')
        if result and result.get('state') == 'complete':
            complete += len(job['episodes'])
            audited_jobs.add(job['id'])
    for worker, job in active.values():
        progress = read(PLAN.parent / 'jobs' / job['id'] / 'attempt1/progress.json') or {}
        if job['id'] not in audited_jobs:
            partial += progress.get('completed', 0)
        rows.append({'worker': worker['name'], 'job': job['id'], 'progress': progress})
    baseline = sum(r['episodes'] for r in manifest['retained'] if r['stage'] == 'test')
    new_saved = complete + partial - baseline
    elapsed = time.time() - started
    rate = new_saved / elapsed if elapsed > 0 and new_saved > 0 else None
    return {'state': 'draining_failure' if failures else 'running', 'heartbeat': time.time(),
        'model': MODEL, 'formal_total': manifest['formal_episodes'],
        'audited_formal': complete, 'active_saved_not_yet_audited': partial,
        'new_saved_since_launch': new_saved, 'active': rows, 'failures': failures,
        'elapsed_seconds': elapsed, 'observed_episodes_per_hour': rate * 3600 if rate else None,
        'provisional_remaining_hours': max(0, manifest['formal_episodes'] - complete - partial) / rate / 3600 if rate else None,
        'eta_note': 'Launch-to-now saved throughput including loading; early batches and patient heterogeneity can strongly bias this estimate.'}


def run(path):
    manifest, plan = load(path)
    control = path.parent
    journal = Journal(control, Path(tempfile.mkdtemp(prefix='medtoolbench-b-qwen-')))
    with FileLease(control / 'supervisor.lock') as owner:
        if (control / 'started.json').exists():
            raise RuntimeError('Explicit reconciliation required before restarting controller')
        old_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            for worker in manifest['workers']:
                validate_worker(plan, worker['name'])
                os.environ['CUDA_VISIBLE_DEVICES'] = str(worker['gpu'])
                with AllocatedGpuLease(worker):
                    pass
        finally:
            if old_cuda is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda
        jobs = {j['id']: j for j in selected(plan, 'test')}
        queue = [jobs[name] for name in manifest['pending']]
        active, failures = {}, []
        started = time.time()
        journal.persist(control / 'started.json', {'pid': os.getpid(), 'time': started,
            'manifest_hash': manifest['manifest_hash'], 'local_journal': str(journal.local)})
        with ThreadPoolExecutor(max_workers=len(manifest['workers'])) as pool:
            while queue or active:
                for future in list(active):
                    if not future.done():
                        continue
                    worker, job = active.pop(future)
                    try:
                        result = future.result()
                        if result['state'] != 'complete':
                            failures.append({'job': job['id'], 'result': result})
                    except Exception:
                        failures.append({'job': job['id'], 'error': traceback.format_exc()})
                    if failures:
                        journal.persist(control / 'needs_review.json', {'failures': failures, 'time': time.time()})
                if not failures and queue and journal.healthy():
                    busy = {w['name'] for w, j in active.values()}
                    for worker in manifest['workers']:
                        if worker['name'] not in busy and queue:
                            job = queue.pop(0)
                            future = pool.submit(attempt, path, job, worker, [owner.fileno()], journal)
                            active[future] = worker, job
                state = snapshot(manifest, plan, active, failures, started)
                state['queued_jobs'] = len(queue)
                journal.put(control / 'status.json', state)
                if failures and not active:
                    state['state'] = 'needs_review'
                    journal.persist(control / 'status.json', state)
                    return
                if queue or active:
                    time.sleep(10)
        retained, pending = inventory(PLAN, plan)
        if pending:
            raise RuntimeError('Qwen B coverage incomplete')
        final_by_id = {r['job']: r for r in retained}
        if any(final_by_id.get(r['job']) != r for r in manifest['retained']):
            raise RuntimeError('Previously certified Qwen evidence changed')
        journal.persist(control / 'complete.json', {'all_pass': True, 'model': MODEL,
            'plan_hash': plan['plan_hash'], 'manifest_hash': manifest['manifest_hash'],
            'formal_episodes': manifest['formal_episodes'], 'jobs': retained,
            'next_model_auto_start': False, 'time': time.time()})
        state.update(state='complete', active=[], queued_jobs=0)
        journal.persist(control / 'status.json', state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, default=CONTROL)
    parser.add_argument('--unit', type=Path, required=True)
    args = parser.parse_args()
    control = args.control.resolve()
    control.mkdir(parents=True, exist_ok=True)
    with (control / 'bootstrap.log').open('x') as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            prepare(control, args.unit.resolve())
            run(control / 'manifest.json')
        except BaseException:
            atomic_json(control / 'bootstrap_failure.json', {'error': traceback.format_exc(), 'time': time.time()})
            raise


if __name__ == '__main__':
    main()
