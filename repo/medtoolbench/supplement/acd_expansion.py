"""User-requested full Llama A/C/D expansion with immutable source references.

Original jobs and completed partial-coverage arms remain unchanged. New B,
Gemma, GPT-OSS and unspecified fourth-model work are never dispatched.
"""
import argparse
import copy
import hashlib
import os
import signal
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import AllocatedGpuLease, FileLease, validate_worker
from . import acd_order, concurrent_pipeline, gptoss_pipeline as engine
from .isolated_execution import model_gate
from .memory_execution import verify_completion
from .runner import load_plan
from .storage_recovery import Journal, atomic_json, execute, read

ROOT = Path(__file__).resolve().parents[3]
CONTROL = ROOT / 'data/acd_llama_expansion_v1'
PRIOR = ROOT / 'data/acd_priority_order_v1'
MODEL = 'llama31_8b'
TEMPLATE = 'qwen35_9b'
CAMPAIGNS = {'a': 'supplement_a_v3', 'c': 'supplement_c_v4', 'd': 'supplement_d_v3'}


def source_hashes():
    files = [Path(__file__).resolve(), Path(__file__).with_name('acd_expansion_runner.py'),
             ROOT / 'repo/tests/test_acd_expansion.py']
    return {str(p): digest(p.read_text()) for p in files}


def scientific_job(job):
    return {k: v for k, v in job.items() if k not in {'id', 'model'}}


def model_artifacts(system):
    path = Path(system['path'])
    index = read(path / 'model.safetensors.index.json')
    if not index or path.name != system['revision']:
        raise RuntimeError('Require the original pinned model snapshot')
    files = [path / name for name in sorted(set(index['weight_map'].values()))]
    if any(not f.is_file() for f in files):
        raise RuntimeError('Model snapshot is incomplete')
    metadata = {str(path / name): hashlib.sha256((path / name).read_bytes()).hexdigest()
                for name in ('config.json', 'tokenizer_config.json', 'model.safetensors.index.json')}
    sizes = {str(f): f.stat().st_size for f in files}
    if sum(sizes.values()) != system['weight_bytes']:
        raise RuntimeError('Model shard sizes differ from the primary registry')
    return {'system': system, 'metadata_sha256': metadata, 'weight_file_bytes': sizes,
            'weight_content_hashes_checked': False, 'no_download': True, 'inference_api': False}


def expanded_plan(parent, cp, system, inputs, sources):
    if system['name'] != MODEL or cp.parent.name not in CAMPAIGNS.values():
        raise RuntimeError('Only the user-requested Llama A/C/D expansion is authorized')
    source_jobs = [j for j in parent['jobs'] if j['model'] == TEMPLATE]
    if not source_jobs or not any(j['stage'] == 'qualification' for j in source_jobs):
        raise RuntimeError('Full template matrix with fresh development is required')
    plan = copy.deepcopy(parent)
    plan.pop('plan_hash')
    plan['systems'] = [copy.deepcopy(system)]
    plan['jobs'] = [dict(copy.deepcopy(j), id='full--' + j['id'].replace(TEMPLATE, MODEL), model=MODEL)
                    for j in source_jobs]
    if any(scientific_job(a) != scientific_job(b) for a, b in zip(source_jobs, plan['jobs'])):
        raise RuntimeError('Expansion changed the scientific assignment')
    plan['created_at'] = time.time()
    plan['session'] = 'medtoolbench-acd-llama-v1'
    plan['source_hashes'].update(sources)
    plan['input_hashes'].update({str(p): digest(p.read_bytes().hex()) for p in inputs})
    plan['require_oracle'] = False
    plan['acd_expansion'] = {'parent_plan': str(cp), 'parent_plan_hash': parent['plan_hash'],
        'template_model': TEMPLATE, 'added_model': MODEL,
        'parent_oracle': str(cp.parent / 'oracle/complete.json') if parent.get('require_oracle') else None,
        'oracle_rationale': 'Identical tasks, fixtures, faults and graders; environment certificate is model-independent',
        'concurrent_transport': 'Previously certified rejection-preserving proxy, only for concurrent arms',
        'selection': 'Explicit user request after primary study; not blinded or preregistered selection',
        'full_matrix_not_adverse_subset': True, 'old_adverse_arm_pooled': False,
        'clinical_accuracy_not_a_gate': True, 'json_envelope_candidate_enabled': False}
    plan['plan_hash'] = digest(plan)
    return plan


def prepare(control, validation):
    if (control / 'manifest.json').exists() or (control / 'campaign_a').exists():
        raise RuntimeError('Never overwrite an expansion or partial preparation')
    proof = read(validation)
    if not proof or not proof.get('all_pass') or proof['source_hashes'] != source_hashes():
        raise RuntimeError('Exact-source complete CPU regression required')
    prior = acd_order.load(PRIOR / 'manifest.json')
    primary_path = ROOT / 'data/primary_study_v1/plan.json'
    primary = read(primary_path)
    system = next(s for s in primary['systems'] if s['name'] == MODEL)
    if system['thinking'] or system['batch_size'] != 4:
        raise RuntimeError('Unexpected primary Llama configuration')
    artifacts = model_artifacts(system)
    artifact_path = control / 'model_artifacts.json'
    if artifact_path.exists():
        raise RuntimeError('Existing artifact record requires review')
    atomic_json(artifact_path, artifacts)
    repair = concurrent_pipeline.load(ROOT / 'data/non_gptoss_priority_v3/manifest.json')
    new_paths, hashes, roles = [], dict(prior['plan_hashes']), {}
    for letter, name in CAMPAIGNS.items():
        cp = ROOT / 'data' / name / 'plan.json'
        parent = load_plan(cp)
        inputs = [cp, primary_path, artifact_path, validation]
        if parent.get('require_oracle'):
            oracle = cp.parent / 'oracle/complete.json'
            certificate = read(oracle)
            if not certificate or not certificate.get('all_pass') or certificate['plan_hash'] != parent['plan_hash']:
                raise RuntimeError('Parent environment validation is incomplete')
            inputs.append(oracle)
        inputs.append(Path(repair['transport_validation']))
        sources = {**source_hashes(), **repair['repair_source_hashes']}
        plan = expanded_plan(parent, cp, system, inputs, sources)
        new_path = control / ('campaign_' + letter) / 'plan.json'
        atomic_json(new_path, plan)
        new_paths.append(str(new_path))
        hashes[str(new_path)] = plan['plan_hash']
        roles[str(cp)] = roles[str(new_path)] = letter.upper()
    manifest = {'created_at': time.time(), 'project': str(ROOT), 'source_hashes': source_hashes(),
        'scheduling_parent': str(PRIOR / 'manifest.json'), 'scheduling_parent_hash': prior['manifest_hash'],
        'old_campaigns': prior['old_campaigns'], 'new_campaigns': new_paths, 'plan_hashes': hashes,
        'campaign_roles': roles, 'dispatch_campaigns': ['A', 'C', 'D'],
        'allowed_models': prior['allowed_models'], 'new_models_added': [MODEL],
        'full_matrix_models': ['qwen35_9b', 'ministral3_8b', MODEL],
        'deferred_models': ['gptoss20b', 'gemma3_12b'], 'retired_models': ['gemma3_12b'],
        'automatic_b_resume': False, 'automatic_deferred_resume': False,
        'execution_manifest': prior['execution_manifest'], 'repair_jobs': prior['repair_jobs'],
        'repair_control': str(ROOT / 'data/non_gptoss_priority_v3'),
        'validation': str(validation), 'validation_hash': digest(proof),
        'model_artifacts': str(artifact_path), 'model_artifacts_hash': digest(artifacts),
        'inference_api': False, 'new_attempts_per_job': 1,
        'original_scientific_plans_unchanged': True,
        'failure_policy': 'No blind retry; runtime failures quarantine the model; integrity failures stop dispatch',
        'scheduling': 'Development first, then existing ACD remainder, then full Llama ACD; old B children drain only'}
    manifest['manifest_hash'] = digest(manifest)
    atomic_json(control / 'manifest.json', manifest)
    load(control / 'manifest.json')


def load(path, require_handoff=False):
    path = Path(path)
    m = read(path)
    wanted = m.pop('manifest_hash')
    if digest(m) != wanted:
        raise RuntimeError('Expansion manifest changed')
    m['manifest_hash'] = wanted
    prior = acd_order.load(Path(m['scheduling_parent']))
    if prior['manifest_hash'] != m['scheduling_parent_hash']:
        raise RuntimeError('Scheduling parent changed')
    for field in ('old_campaigns', 'allowed_models', 'execution_manifest', 'repair_jobs'):
        if m[field] != prior[field]:
            raise RuntimeError('Original scope changed: ' + field)
    if (m['new_models_added'] != [MODEL] or m['automatic_b_resume'] or m['automatic_deferred_resume']
            or m['new_attempts_per_job'] != 1 or m['source_hashes'] != source_hashes()):
        raise RuntimeError('Undeclared model, policy, or source change')
    if digest(read(m['validation'])) != m['validation_hash']:
        raise RuntimeError('Validation changed')
    if digest(read(m['model_artifacts'])) != m['model_artifacts_hash']:
        raise RuntimeError('Model artifact evidence changed')
    if len(m['new_campaigns']) != 3 or {m['campaign_roles'][p] for p in m['new_campaigns']} != {'A', 'C', 'D'}:
        raise RuntimeError('Incomplete or unexpected new campaign scope')
    if require_handoff:
        h = read(path.parent / 'handoff.json')
        if not h or h.get('manifest_hash') != wanted:
            raise RuntimeError('Ownership handoff missing')
        checksum = h.pop('handoff_hash')
        if digest(h) != checksum:
            raise RuntimeError('Ownership receipt changed')
    return m


def dispatchable(cp, manifest):
    return str(cp) in manifest['campaign_roles']


def folder_for(control, cp, job, manifest):
    if concurrent_pipeline.repair_key(cp, job['id']) in manifest['repair_jobs']:
        return Path(manifest['repair_control']) / 'repair_jobs' / job['id']
    return cp.parent / 'jobs' / job['id']


def inventory(control, plans, manifest):
    pending, blocked, complete = [], {}, 0
    for cp, plan in plans:
        if not dispatchable(cp, manifest):
            continue
        for job in plan['jobs']:
            if job['model'] not in manifest['allowed_models']:
                continue
            result = read(folder_for(control, cp, job, manifest) / 'result.json')
            if result and result.get('state') == 'complete':
                complete += len(job['episodes'])
            else:
                pending.append({'plan': str(cp), 'job': job['id'], 'model': job['model'],
                    'stage': job['stage'], 'episodes': len(job['episodes'])})
                if result:
                    blocked[job['model']] = result
    return pending, blocked, complete


def eligible(control, plans, manifest, active, blocked):
    ready = []
    for index, (cp, plan) in enumerate(plans):
        if not dispatchable(cp, manifest):
            continue
        gates = {}
        for job in plan['jobs']:
            if (job['model'] not in manifest['allowed_models'] or job['model'] in blocked
                    or (str(cp), job['id']) in active):
                continue
            folder = folder_for(control, cp, job, manifest)
            if (folder / 'result.json').exists() or list(folder.glob('attempt[0-9]*')):
                continue
            if job['stage'] == 'test':
                if job['model'] not in gates:
                    gates[job['model']] = model_gate(cp, plan, job['model'])
                if gates[job['model']] != 'passed':
                    continue
            ready.append((0 if job['stage'] == 'qualification' else 1,
                int(str(cp) in manifest['new_campaigns']), index, cp, plan, job))
    return ready


def attempt(path, cp, job, worker, journal, inherited):
    manifest = load(path, require_handoff=True)
    if not dispatchable(cp, manifest) or job['model'] not in manifest['allowed_models']:
        raise RuntimeError('Job outside declared ACD scope')
    if str(cp) not in manifest['new_campaigns']:
        # Original inference wrapper and original execution manifest are retained.
        return engine.attempt(path, cp, job, worker, journal, inherited)
    if job['model'] != MODEL:
        raise RuntimeError('Only Llama is authorized for the new matrix')
    plan = load_plan(cp)
    folder = folder_for(path.parent, cp, job, manifest)
    with FileLease(cp.parent / 'locks' / (job['id'] + '.lock')) as lease:
        if (folder / 'result.json').exists() or list(folder.glob('attempt[0-9]*')):
            raise RuntimeError('Never overwrite or blindly retry expansion evidence')
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / 'attempt1'
        cmd = [plan['python'], '-u', '-m', 'medtoolbench.supplement.acd_expansion_runner',
            '--expansion-manifest', str(path), '--plan', str(cp), '--job', job['id'],
            '--worker', worker['name'], '--out', str(out)]
        journal.persist(folder / 'command.json', {'command': cmd, 'worker': worker,
            'manifest_hash': manifest['manifest_hash'], 'plan_hash': plan['plan_hash']})
        result = execute(cmd, folder / 'attempt1.log', worker, path.parent, [*inherited, lease.fileno()], journal)
        record = {'job': job['id'], 'model': job['model'], 'worker': worker['name'], 'run': str(out),
            'attempts': [result], 'manifest_hash': manifest['manifest_hash']}
        if result['returncode'] == 0 and (out / 'complete.json').exists():
            proof = verify_completion(out, plan, job)
            if read(out / 'config.json').get('acd_expansion', {}).get('manifest_hash') != manifest['manifest_hash']:
                raise RuntimeError('Missing expansion execution identity')
            record.update(state='complete', episodes=proof['episodes'])
        else:
            if (out / 'audit.json').exists() or (out / 'complete.json').exists():
                raise RuntimeError('Audit-invalid evidence cannot be accepted or retried')
            record.update(state='quarantined', episodes=0, assigned_episodes=len(job['episodes']),
                error=read(out / 'failure.json'))
        journal.persist(folder / 'result.json', record)
        return record


def handoff(path):
    m = load(path)
    control = path.parent
    if (control / 'handoff.json').exists():
        raise RuntimeError('Never overwrite handoff evidence')
    old = Path(m['scheduling_parent']).parent
    state = read(old / 'status.json')
    pid = state['pid']
    command = engine.process_cmdline(pid)
    if '-m medtoolbench.supplement.acd_order run' not in command:
        raise RuntimeError('Unexpected previous dispatcher')
    os.kill(pid, signal.SIGSTOP)
    try:
        state = read(old / 'status.json')
        children = []
        for name, (cp, job_id) in state['busy'].items():
            row = read(old / 'workers' / (name + '.json'))
            config = read(Path(row['out']) / 'config.json')
            plan = load_plan(Path(cp))
            if config['plan_hash'] != plan['plan_hash'] or config['job']['id'] != job_id:
                raise RuntimeError('Child launch changing; resume and retry later')
            children.append({**row, 'config': config})
        previous = read(old / 'handoff.json')
        children.extend(r for r in previous['live_children_preserved'] if r['pid'] in state['preserved_children'])
        if len({r['worker']['name'] for r in children}) != len(children):
            raise RuntimeError('Duplicate GPU ownership')
        for row in children:
            current = engine.process_cmdline(row['pid'])
            if current and (row['out'] not in current or '-m medtoolbench.supplement.isolated_runner ' not in current):
                raise RuntimeError('Unexpected child identity')
            row['observed_cmdline'] = current
        snapshot = {'time': time.time(), 'manifest_hash': m['manifest_hash'], 'old_controller_pid': pid,
            'old_controller_cmdline': command, 'old_status': state, 'live_children_preserved': children,
            'no_model_children_signalled': True}
        snapshot['handoff_hash'] = digest(snapshot)
        atomic_json(control / 'handoff.json', snapshot)
        load(path, require_handoff=True)
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
        raise RuntimeError('Previous dispatcher did not exit')
    atomic_json(old / 'superseded_llama_expansion.json', {'successor': str(control), 'time': time.time(),
        'original_results_and_live_model_children_preserved': True})


def finalize(control, plans, manifest):
    from .report import generate
    references = []
    for letter in ('A', 'C', 'D'):
        selected = [(cp, p) for cp, p in plans if manifest['campaign_roles'].get(str(cp)) == letter]
        view = copy.deepcopy(selected[0][1])
        view.pop('plan_hash')
        view['jobs'], view['systems'] = [], []
        systems = {}
        resolved = []
        for cp, plan in selected:
            systems.update({s['name']: s for s in plan['systems'] if s['name'] in manifest['allowed_models']})
            for job in plan['jobs']:
                if job['model'] not in manifest['allowed_models']:
                    continue
                file = folder_for(control, cp, job, manifest) / 'result.json'
                r = read(file)
                if not r or r.get('state') != 'complete':
                    raise RuntimeError('Full ACD reporting requires every assigned job')
                verify_completion(Path(r['run']), plan, job)
                resolved.append((job, file, r, plan['plan_hash']))
                view['jobs'].append(job)
        if len({j['id'] for j in view['jobs']}) != len(view['jobs']):
            raise RuntimeError('Overlapping analysis job identifiers')
        view.update(systems=list(systems.values()), analysis_only=True, require_oracle=False,
            source_plans={str(cp): p['plan_hash'] for cp, p in selected},
            full_matrix_models=manifest['full_matrix_models'], acd_expansion_manifest=str(control / 'manifest.json'))
        view['source_hashes'].update(manifest['source_hashes'])
        view['plan_hash'] = digest(view)
        vp = control / ('combined_' + letter.lower()) / 'plan.json'
        if vp.exists():
            raise RuntimeError('Partial combined analysis requires explicit reconciliation')
        atomic_json(vp, view)
        for job, source, result, plan_hash in resolved:
            ref = {**result, 'analysis_reference_only': True, 'source_result': str(source),
                'source_result_hash': digest(result), 'source_plan_hash': plan_hash}
            atomic_json(vp.parent / 'jobs' / job['id'] / 'result.json', ref)
            references.append({'plan': str(vp), 'job': job['id'], 'stage': job['stage'],
                               'episodes': len(job['episodes']), 'source': str(source)})
        generate(vp)
    atomic_json(control / 'complete.json', {'all_pass': True, 'time': time.time(),
        'manifest_hash': manifest['manifest_hash'], 'full_matrix_models': manifest['full_matrix_models'],
        'formal_episodes': sum(r['episodes'] for r in references if r['stage'] == 'test'),
        'development_episodes': sum(r['episodes'] for r in references if r['stage'] == 'qualification'),
        'jobs': references, 'scope': 'Three-model ACD plus retained adverse-only arms',
        'entire_project_complete': False, 'automatic_b_resume': False})


def run(path):
    manifest = load(path, require_handoff=True)
    engine.load = load
    control = path.parent
    snapshot = read(control / 'handoff.json')
    if engine.process_cmdline(snapshot['old_controller_pid']):
        raise RuntimeError('Previous dispatcher still exists')
    plans = [(Path(raw), load_plan(Path(raw))) for raw in manifest['old_campaigns'] + manifest['new_campaigns']]
    if any(p['plan_hash'] != manifest['plan_hashes'][str(cp)] for cp, p in plans):
        raise RuntimeError('Scientific plan identity changed')
    workers = plans[0][1]['workers']
    for w in workers:
        validate_worker(plans[0][1], w['name'])
    journal = Journal(control, Path(tempfile.mkdtemp(prefix='medtoolbench-acd-llama-')))
    remaining, busy, fatal, cursor = list(snapshot['live_children_preserved']), {}, None, 0
    with FileLease(control / 'supervisor.lock') as owner, ThreadPoolExecutor(max_workers=8) as pool:
        if (control / 'started.json').exists():
            raise RuntimeError('Restart requires explicit ownership reconciliation')
        journal.persist(control / 'started.json', {'pid': os.getpid(), 'time': time.time(),
            'manifest_hash': manifest['manifest_hash']})
        while True:
            try:
                load(path, require_handoff=True)
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
                        candidates = eligible(control, plans, manifest, active, blocked)
                        if not candidates:
                            continue
                        candidates.sort(key=lambda r: (r[0], r[1], (r[2] - cursor) % len(plans), r[5]['id']))
                        _, _, index, cp, plan, job = candidates[0]
                        cursor = (index + 1) % len(plans)
                        key = (str(cp), job['id'])
                        busy[worker['name']] = (pool.submit(attempt, path, cp, job, worker, journal,
                            [owner.fileno()]), key)
                        active.add(key)
                journal.put(control / 'pending.json', {'jobs': pending, 'time': time.time(),
                    'completed_including_development': completed, 'b_deferred_not_omitted_from_project': True})
                journal.put(control / 'status.json', {'state': 'integrity_stop' if fatal else 'running',
                    'pid': os.getpid(), 'busy': {n: list(k) for n, (_, k) in busy.items()},
                    'preserved_children': [r['pid'] for r in remaining], 'blocked_models': list(blocked),
                    'full_matrix_models': manifest['full_matrix_models'], 'automatic_b_resume': False,
                    'heartbeat': time.time(), 'manifest_hash': manifest['manifest_hash']})
                if not busy and not remaining:
                    if fatal or pending:
                        journal.persist(control / 'needs_review.json', {'error': fatal,
                            'blocked_models': blocked, 'pending_jobs': len(pending), 'time': time.time()})
                    else:
                        journal.persist(control / 'inference_complete.json', {'all_pass': True,
                            'time': time.time(), 'scope': 'ACD inference and shard audit, reports pending'})
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
    p.add_argument('command', choices=['prepare', 'handoff', 'run'])
    p.add_argument('--control', type=Path, default=CONTROL)
    p.add_argument('--validation', type=Path)
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args.control.resolve(), args.validation.resolve())
    elif args.command == 'handoff':
        handoff(args.control.resolve() / 'manifest.json')
    else:
        run(args.control.resolve() / 'manifest.json')
