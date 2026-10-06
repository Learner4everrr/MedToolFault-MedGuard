"""Audit-only recovery over identical native A/C/D plans and saved inference.

Original source files, plans, failed receipts and saved episodes are immutable.
The successor has byte-equivalent scientific plans and fresh result overlays.
Only never-started jobs can receive inference; all 260 development audits gate
formal work. No clinical outcome threshold or automatic inference retry.
"""
import copy
import os
import time
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import FileLease
from . import gptoss_native_pipeline as native
from .gptoss_replay_audit import REVISION, audit_native, checksum, evidence_hashes, tokenizer_for
from .memory_execution import verify_completion
from .runner import load_plan
from .storage_recovery import atomic_json, read, execute

ROOT = native.ROOT
CONTROL = ROOT / 'data/gptoss_native_replay_recovery_v1'
PARENT = native.CONTROL / 'manifest.json'
PARENT_LOAD, PARENT_FINALIZE = native.load, native.finalize


def source_hashes():
    paths = [Path(__file__).with_name(n + '.py') for n in
             ('gptoss_replay_audit', 'gptoss_replay_recovery', 'gptoss_replay_runner')]
    paths += [ROOT / 'repo/scripts' / n for n in ('validate_gptoss_replay.py', 'run_gptoss_replay.py')]
    paths.append(ROOT / 'repo/tests/test_gptoss_replay.py')
    return {str(p): digest(p.read_text()) for p in paths}


def process_present(pid):
    return bool(pid) and Path('/proc/' + str(pid)).exists()


def stopped(parent):
    state = read(parent.parent / 'status.json')
    if not state or state.get('busy') or not (parent.parent / 'needs_review.json').exists():
        raise RuntimeError('Original controller must have stopped and drained')
    if process_present(state.get('pid')):
        raise RuntimeError('Original controller PID still exists; reconcile ownership first')
    for f in (parent.parent / 'workers').glob('*.json'):
        if process_present(read(f).get('pid')):
            raise RuntimeError('Original worker PID still exists')


def recovery_eligible(folder, plan, job):
    result = read(folder / 'result.json')
    attempts = sorted(p for p in folder.glob('attempt[0-9]*') if p.is_dir())
    if not result:
        if attempts:
            raise RuntimeError('Partial/unresolved inference cannot be adopted or restarted')
        return None
    if job['stage'] != 'qualification' or job['model'] != native.MODEL or len(attempts) != 1:
        raise RuntimeError('Only the completed native development inference is recoverable')
    out = Path(result['run'])
    if out.resolve() != attempts[0].resolve() or (out / 'failure.json').exists():
        raise RuntimeError('Not an audit-only failure')
    verify_completion(out, plan, job)
    if result['state'] == 'quarantined':
        log = (folder / 'attempt1.native.log').read_text()
        if ('RuntimeError: Native input/history replay mismatch:' not in log or
                len(result['attempts']) != 1 or result['attempts'][0] != {'returncode': 1, 'timeout': None}):
            raise RuntimeError('Unreviewed infrastructure failure cannot be recovered')
    elif result['state'] == 'complete':
        if not read(out / 'native_audit.json').get('all_pass'):
            raise RuntimeError('Missing original successful native certificate')
    else:
        raise RuntimeError('Unexpected historical result state')
    return result


def prepare(control, unit, parent=PARENT):
    control, unit, parent = Path(control).resolve(), Path(unit).resolve(), Path(parent).resolve()
    if (control / 'manifest.json').exists() or (control / 'campaign_a').exists():
        raise RuntimeError('Never overwrite recovery preparation')
    sources = source_hashes()
    proof = read(unit)
    if (not proof or not proof.get('all_pass') or proof.get('source_hashes') != sources or
            proof.get('native_source_hashes') != native.source_hashes() or proof.get('held_out_tasks_used') is not False):
        raise RuntimeError('Exact-source full regression receipt required')
    base = PARENT_LOAD(parent)
    with FileLease(parent.parent / 'supervisor.lock'):
        stopped(parent)
        campaigns, roles, hashes, origins, adopted = [], {}, {}, {}, []
        total = generations = 0
        tokenizer = None
        for raw in base['campaigns']:
            cp = Path(raw); plan = load_plan(cp)
            if plan['plan_hash'] != base['plan_hashes'][raw]:
                raise RuntimeError('Original native plan identity changed')
            letter = base['campaign_roles'][raw]
            dest = control / ('campaign_' + letter.lower()) / 'plan.json'
            # Same plan object/hash, not a new scientific assignment or protocol.
            atomic_json(dest, plan)
            if read(dest) != read(cp):
                raise RuntimeError('Recovery changed a scientific plan')
            campaigns.append(str(dest)); roles[str(dest)] = letter; hashes[str(dest)] = plan['plan_hash']
            origins[str(dest)] = str(cp)
            for job in plan['jobs']:
                folder = cp.parent / 'jobs' / job['id']
                with FileLease(cp.parent / 'locks' / (job['id'] + '.lock')):
                    original = recovery_eligible(folder, plan, job)
                    if original is None:
                        continue
                    out = Path(original['run'])
                    if tokenizer is None:
                        tokenizer = tokenizer_for(plan)
                    receipt = control / 'replay_audits' / letter / (job['id'] + '.json')
                    corrected = audit_native(out, plan, receipt, tokenizer)
                    record = {**copy.deepcopy(original), 'state': 'complete', 'episodes': corrected['episodes'],
                        'replay_audit_path': str(receipt), 'replay_audit_hash': digest(corrected),
                        'audit_recovery_revision': REVISION, 'inference_reused': True,
                        'original_result': str(folder / 'result.json'), 'original_result_hash': digest(original),
                        'original_result_state': original['state'], 'original_failed_receipt_preserved': True,
                        'scientific_plan_unchanged': True}
                    target = dest.parent / 'jobs' / job['id'] / 'result.json'
                    atomic_json(target, record)
                    adopted.append({'result': str(target), 'result_hash': digest(record),
                        'original_result': str(folder / 'result.json'), 'original_result_hash': digest(original),
                        'original_log': str(folder / 'attempt1.native.log'),
                        'original_log_sha256': checksum(folder / 'attempt1.native.log'),
                        'replay_audit': str(receipt), 'replay_audit_hash': digest(corrected)})
                    total += corrected['episodes']; generations += corrected['generations']
                    print('Re-audited', letter, job['id'], corrected['episodes'], flush=True)
        if (total, generations, len(adopted)) != (230, 654, 9):
            raise RuntimeError('Recovery scope differs from the reviewed 230-episode incident')
        if sources != source_hashes():
            raise RuntimeError('Recovery implementation changed during validation')
        m = copy.deepcopy(base); m.pop('manifest_hash')
        m.update(campaigns=campaigns, campaign_roles=roles, plan_hashes=hashes,
            replay_recovery_revision=REVISION, replay_source_hashes=sources,
            parent_native_manifest=str(parent), parent_native_manifest_hash=base['manifest_hash'],
            original_campaigns=origins, adopted_results=adopted, created_at=time.time(),
            recovered_episodes=total, recovered_generations=generations, inference_reexecuted=0,
            remaining_development_episodes=30, inference_and_scientific_protocol_changed=False,
            session='medtoolbench-gptoss-replay-v1')
        m['proofs'][str(unit)] = digest(proof)
        m['manifest_hash'] = digest(m)
        atomic_json(control / 'manifest.json', m)
        load(control / 'manifest.json')


def load(path):
    path = Path(path); m = read(path)
    if not m:
        raise RuntimeError('Replay recovery manifest missing')
    expected = m.pop('manifest_hash')
    if digest(m) != expected or m['replay_source_hashes'] != source_hashes() or m['replay_recovery_revision'] != REVISION:
        raise RuntimeError('Replay recovery manifest/source changed')
    m['manifest_hash'] = expected
    parent = PARENT_LOAD(Path(m['parent_native_manifest']))
    if parent['manifest_hash'] != m['parent_native_manifest_hash']:
        raise RuntimeError('Parent native cohort changed')
    for key in ('source_hashes', 'version', 'model', 'automatic_b_resume', 'gemma_retired', 'old_gptoss_pooled',
                'clinical_accuracy_gate', 'all_development_before_formal', 'max_attempts_per_job',
                'formal_episodes', 'development_episodes', 'inference_api'):
        if m[key] != parent[key]:
            raise RuntimeError('Audit repair changed inference/scope: ' + key)
    if m['inference_and_scientific_protocol_changed'] or m['inference_reexecuted'] != 0:
        raise RuntimeError('Inference changes are not authorized by this audit amendment')
    if len(m['campaigns']) != 3 or set(m['original_campaigns'].values()) != set(parent['campaigns']):
        raise RuntimeError('Only original A/C/D scope may continue')
    for cp in m['campaigns']:
        original = m['original_campaigns'][cp]
        if (m['plan_hashes'][cp] != parent['plan_hashes'][original] or
                m['campaign_roles'][cp] != parent['campaign_roles'][original]):
            raise RuntimeError('Recovery plan mapping changed')
    for file, checksum_ in m['proofs'].items():
        if digest(read(file)) != checksum_:
            raise RuntimeError('Replay proof changed: ' + file)
    for item in m['adopted_results']:
        for name, hash_name in (('result', 'result_hash'), ('original_result', 'original_result_hash'), ('replay_audit', 'replay_audit_hash')):
            if digest(read(item[name])) != item[hash_name]:
                raise RuntimeError('Recovery/adoption receipt changed')
    return m


def verify_result(result, plan, job, full=False):
    if not result or result.get('state') != 'complete':
        raise RuntimeError('Missing complete replay-audited result')
    out = Path(result['run'])
    verify_completion(out, plan, job)
    proof = read(result['replay_audit_path'])
    if (not proof or not proof.get('all_pass') or proof.get('revision') != REVISION or
            proof.get('plan_hash') != plan['plan_hash'] or proof.get('job') != job['id'] or
            proof.get('episodes') != len(job['episodes']) or proof.get('run') != str(out.resolve()) or
            digest(proof) != result['replay_audit_hash']):
        raise RuntimeError('Corrected native audit identity/coverage failed')
    if full and evidence_hashes(out) != proof['evidence_sha256']:
        raise RuntimeError('Certified raw evidence changed')
    return proof


def development_passed(manifest):
    total = 0
    for raw in manifest['campaigns']:
        cp = Path(raw); plan = load_plan(cp)
        for job in plan['jobs']:
            if job['stage'] != 'qualification':
                continue
            result = read(cp.parent / 'jobs' / job['id'] / 'result.json')
            if not result or result.get('state') != 'complete':
                return False
            verify_result(result, plan, job)
            total += len(job['episodes'])
    return total == 260


def authorize_job(manifest, cp, plan, job, allow_launch_log=False):
    if str(cp) not in manifest['campaigns'] or job['model'] != native.MODEL or job not in plan['jobs']:
        raise RuntimeError('Job outside recovered GPT-OSS A/C/D scope')
    if plan['plan_hash'] != manifest['plan_hashes'][str(cp)]:
        raise RuntimeError('Recovery plan identity mismatch')
    old = Path(manifest['original_campaigns'][str(cp)]).parent / 'jobs' / job['id']
    folder = cp.parent / 'jobs' / job['id']
    started = [p for p in folder.glob('attempt[0-9]*')
               if p.is_dir() or not allow_launch_log or p.name != 'attempt1.replay.log']
    if ((folder / 'result.json').exists() or started or
            (old / 'result.json').exists() or list(old.glob('attempt[0-9]*'))):
        raise RuntimeError('Previously started or recovered inference cannot run again')
    if job['stage'] == 'test' and not development_passed(manifest):
        raise RuntimeError('All 260 development audits must precede formal work')


def attempt(path, cp, job, worker, journal, inherited):
    m, plan = load(path), load_plan(cp)
    folder = cp.parent / 'jobs' / job['id']
    with FileLease(cp.parent / 'locks' / (job['id'] + '.lock')) as lease:
        authorize_job(m, cp, plan, job)
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / 'attempt1'
        cmd = [plan['python'], '-u', '-m', 'medtoolbench.supplement.gptoss_replay_runner',
            '--replay-manifest', str(path), '--plan', str(cp), '--job', job['id'], '--worker', worker['name'], '--out', str(out)]
        journal.persist(folder / 'command.json', {'command': cmd, 'worker': worker, 'manifest_hash': m['manifest_hash']})
        outcome = execute(cmd, folder / 'attempt1.replay.log', worker, path.parent, [*inherited, lease.fileno()], journal)
        result = {'job': job['id'], 'model': native.MODEL, 'worker': worker['name'], 'run': str(out),
            'attempts': [outcome], 'manifest_hash': m['manifest_hash'], 'inference_reused': False}
        proof = read(out / 'native_replay_audit.json')
        if outcome['returncode'] == 0 and proof and proof.get('all_pass'):
            result.update(state='complete', episodes=proof['episodes'],
                replay_audit_path=str(out / 'native_replay_audit.json'), replay_audit_hash=digest(proof))
            verify_result(result, plan, job, full=True)
        else:
            result.update(state='quarantined', episodes=0, assigned_episodes=len(job['episodes']),
                error=read(out / 'native_audit_failure.json') or read(out / 'failure.json'),
                reason='Execution or independent audit failure; no automatic retries')
        journal.persist(folder / 'result.json', result)
        return result


def finalize(control, manifest, plans):
    for cp, plan in plans:
        for job in plan['jobs']:
            verify_result(read(cp.parent / 'jobs' / job['id'] / 'result.json'), plan, job, full=True)
    PARENT_FINALIZE(control, {**manifest, 'source_hashes': {**manifest['source_hashes'], **manifest['replay_source_hashes']}}, plans)
    atomic_json(control / 'replay_recovery_complete.json', {'all_pass': True, 'revision': REVISION,
        'manifest_hash': manifest['manifest_hash'], 'recovered_development_without_inference': 230,
        'new_development_inference': 30, 'formal_episodes': 17967, 'old_evidence_preserved': True,
        'entire_project_complete': False, 'time': time.time()})


def run(path):
    path = Path(path); manifest = load(path)
    for raw in manifest['campaigns']:
        cp = Path(raw); plan = load_plan(cp)
        for job in plan['jobs']:
            result = read(cp.parent / 'jobs' / job['id'] / 'result.json')
            if result:
                verify_result(result, plan, job, full=True)
    for item in manifest['adopted_results']:
        if checksum(item['original_log']) != item['original_log_sha256']:
            raise RuntimeError('Historical failure log changed')
    with FileLease(Path(manifest['parent_native_manifest']).parent / 'supervisor.lock'):
        stopped(Path(manifest['parent_native_manifest']))
        # In-process adapter only. Frozen native sources and scientific plans
        # stay unchanged; all children load this exact-source amendment.
        native.load, native.attempt = load, attempt
        native.development_passed, native.finalize = development_passed, finalize
        native.run(path)
