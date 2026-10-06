"""Additive compiler/OOM amendment over immutable native/replay evidence.

Reuse all 5,692 audited formal and 260 development episodes. The two known
OOM-incomplete shards get one explicit new attempt, independent of outcomes.
Their 80 partial episodes are archived, never pooled or selectively retained.
All other unstarted scientific assignments remain byte-equivalent.
"""
import copy
import time
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import FileLease
from . import gptoss_native_pipeline as native
from . import gptoss_replay_recovery as replay
from .gptoss_compile import REVISION, RECOMPILE_LIMIT, ACCUMULATED_LIMIT
from .gptoss_replay_audit import checksum, evidence_hashes
from .runner import load_plan
from .storage_recovery import atomic_json, read, execute

ROOT = native.ROOT
CONTROL = ROOT / 'data/gptoss_compile_recovery_v1'
PARENT = replay.CONTROL / 'manifest.json'
PARENT_LOAD = replay.load
PARENT_FINALIZE = native.finalize
OOM_JOBS = {'native--matched_controls--gptoss20b--00': 38,
            'native--matched_controls--gptoss20b--01': 42}


def source_hashes():
    paths = [Path(__file__).with_name(n + '.py') for n in
             ('gptoss_compile', 'gptoss_compile_recovery', 'gptoss_compile_runner')]
    paths += [ROOT / 'repo/scripts' / n for n in
              ('validate_gptoss_compile.py', 'validate_gptoss_compile_gpu.py', 'run_gptoss_compile.py')]
    paths.append(ROOT / 'repo/tests/test_gptoss_compile.py')
    return {str(p): digest(p.read_text()) for p in paths}


def validate_oom(folder, plan, job):
    result = read(folder / 'result.json')
    if (job['id'] not in OOM_JOBS or job['stage'] != 'test' or job['model'] != native.MODEL or
            not result or result.get('state') != 'quarantined'):
        raise RuntimeError('Only the two reviewed native OOM shards qualify')
    attempts = sorted(p for p in folder.glob('attempt[0-9]*') if p.is_dir())
    out = folder / 'attempt1'
    if (attempts != [out] or Path(result['run']).resolve() != out.resolve() or
            result['attempts'] != [{'returncode': 1, 'timeout': None}]):
        raise RuntimeError('Unexpected previous attempt history')
    if any((out / f).exists() for f in ('audit.json', 'complete.json', 'native_replay_audit.json')):
        raise RuntimeError('Audited/complete inference cannot be selected for a retry')
    failure = read(out / 'failure.json') or {}
    error = failure.get('error', '')
    log = folder / 'attempt1.replay.log'
    text = log.read_text()
    if ('torch.OutOfMemoryError: CUDA out of memory' not in error or
            'math_attention' not in error or 'hit config.recompile_limit (8)' not in text):
        raise RuntimeError('Failure is not the reviewed compiler-fallback OOM')
    config = read(out / 'config.json')
    if config['plan_hash'] != plan['plan_hash'] or config['job'] != job:
        raise RuntimeError('Failed assignment changed')
    count = len(list((out / 'episodes').glob('*.json.gz')))
    if count != OOM_JOBS[job['id']] or (read(out/'progress.json') or {}).get('completed') != count:
        raise RuntimeError('Unexpected partial evidence coverage')
    return {'run': str(out.resolve()), 'evidence_sha256': evidence_hashes(out),
        'log': str(log.resolve()), 'log_sha256': checksum(log), 'partial_episodes_excluded': count,
        'reason': 'Compiler fallback CUDA OOM; one outcome-independent infrastructure retry',
        'prior_partial_rows_pooled': False}


def validate_proofs(unit, gpu):
    sources = source_hashes()
    u, g = read(unit), read(gpu)
    if (not u or not u.get('all_pass') or u.get('source_hashes') != sources or
            u.get('native_source_hashes') != native.source_hashes() or
            u.get('replay_source_hashes') != replay.source_hashes() or u.get('held_out_tasks_used') is not False):
        raise RuntimeError('Exact-source complete regression receipt required')
    gpu_sources = {str(ROOT/'repo/medtoolbench/supplement/gptoss_compile.py'),
                   str(ROOT/'repo/scripts/validate_gptoss_compile_gpu.py')}
    if (not g or not g.get('all_pass') or g.get('held_out_tasks_used') is not False or
            g.get('counts_toward_results') is not False or g.get('dense_fallback_calls') != 0 or
            len(g.get('attention_checks', [])) != 28 or g.get('model_stress_cases') != 6 or
            g.get('max_context') != 16384 or g.get('max_generation_budget') != 4096 or
            g.get('source_hashes') != {p: sources[p] for p in gpu_sources}):
        raise RuntimeError('Exact-source long-context GPU validation required')
    settings = g.get('repair', {})
    if (settings.get('revision') != REVISION or settings.get('recompile_limit') != RECOMPILE_LIMIT or
            settings.get('accumulated_recompile_limit') != ACCUMULATED_LIMIT or
            settings.get('fail_on_recompile_limit_hit') is not True or settings.get('suppress_errors') is not False):
        raise RuntimeError('Unvalidated compiler configuration')
    return sources


def prepare(control, unit, gpu, parent=PARENT):
    control, unit, gpu, parent = [Path(x).resolve() for x in (control, unit, gpu, parent)]
    if (control/'manifest.json').exists() or (control/'campaign_a').exists():
        raise RuntimeError('Never overwrite compiler recovery preparation')
    sources = validate_proofs(unit, gpu)
    base = PARENT_LOAD(parent)
    campaigns, roles, hashes, origins, history, adopted, retries = [], {}, {}, {}, [], [], []
    counts = {'qualification': 0, 'test': 0}
    with FileLease(parent.parent/'supervisor.lock'):
        replay.stopped(parent)
        for raw in base['campaigns']:
            cp = Path(raw); plan = load_plan(cp); letter = base['campaign_roles'][raw]
            if plan['plan_hash'] != base['plan_hashes'][raw]:
                raise RuntimeError('Parent scientific plan changed')
            dest = control/('campaign_'+letter.lower())/'plan.json'
            atomic_json(dest, plan)
            if read(dest) != read(cp):
                raise RuntimeError('Scientific assignments changed')
            campaigns.append(str(dest)); roles[str(dest)] = letter; hashes[str(dest)] = plan['plan_hash']
            origins[str(dest)] = str(cp)
            for job in plan['jobs']:
                folder = cp.parent/'jobs'/job['id']
                oldpath = folder/'result.json'; original = read(oldpath)
                item = {'campaign': str(dest), 'job': job['id'], 'original_result': str(oldpath),
                    'original_result_hash': digest(original) if original else None}
                if original and original.get('state') == 'complete':
                    replay.verify_result(original, plan, job, full=True)
                    row = {**copy.deepcopy(original), 'inference_reused': True,
                        'compiler_recovery_adopted': True, 'compiler_recovery_revision': REVISION,
                        'original_result': str(oldpath), 'original_result_hash': digest(original)}
                    target = dest.parent/'jobs'/job['id']/'result.json'
                    atomic_json(target, row)
                    adopted.append({'result': str(target), 'hash': digest(row)})
                    item['policy'] = 'reuse_complete'
                    counts[job['stage']] += len(job['episodes'])
                elif original:
                    authorization = validate_oom(folder, plan, job)
                    item.update(policy='one_oom_retry', authorization=authorization)
                    retries.append(job['id'])
                else:
                    if list(folder.glob('attempt[0-9]*')):
                        raise RuntimeError('Unresolved partial inference needs separate review')
                    item['policy'] = 'unstarted'
                history.append(item)
        if counts != {'qualification':260, 'test':5692} or set(retries) != set(OOM_JOBS) or len(adopted) != 58:
            raise RuntimeError('Incident scope differs from reviewed completion and failure counts')
        if sources != source_hashes():
            raise RuntimeError('Sources changed during preparation')
        m = copy.deepcopy(base); m.pop('manifest_hash')
        m.update(campaigns=campaigns, campaign_roles=roles, plan_hashes=hashes, original_campaigns=origins,
            parent_replay_manifest=str(parent), parent_replay_manifest_hash=base['manifest_hash'],
            compiler_recovery_revision=REVISION, compiler_source_hashes=sources,
            compiler_gpu_validation=str(gpu), compiler_unit_validation=str(unit),
            compiler_history=history, compiler_adopted=adopted, compiler_configuration_changed=True,
            clinical_protocol_changed=False, physical_batch_size=2, extra_attempts_per_oom_job=1,
            automatic_retry=False, completed_inference_rerun=False, prior_partial_rows_pooled=False,
            archived_partial_episodes=80, reused_formal_episodes=5692, reused_development_episodes=260,
            new_formal_assignments=12275, new_development_assignments=0,
            recovered_episodes=5952, remaining_development_episodes=0, created_at=time.time(),
            session='medtoolbench-gptoss-compile-v1')
        m['proofs'][str(unit)] = digest(read(unit)); m['proofs'][str(gpu)] = digest(read(gpu))
        m['manifest_hash'] = digest(m)
        atomic_json(control/'manifest.json', m)
        load(control/'manifest.json')


def load(path):
    m = read(path)
    if not m:
        raise RuntimeError('Compiler recovery manifest missing')
    wanted = m.pop('manifest_hash')
    if digest(m) != wanted or m.get('compiler_source_hashes') != source_hashes() or m.get('compiler_recovery_revision') != REVISION:
        raise RuntimeError('Compiler recovery source/manifest changed')
    m['manifest_hash'] = wanted
    parent = PARENT_LOAD(Path(m['parent_replay_manifest']))
    if parent['manifest_hash'] != m['parent_replay_manifest_hash']:
        raise RuntimeError('Original replay cohort changed')
    for key in ('source_hashes', 'version', 'model', 'formal_episodes', 'development_episodes',
                'inference_api', 'automatic_b_resume', 'gemma_retired', 'old_gptoss_pooled', 'clinical_accuracy_gate'):
        if m[key] != parent[key]:
            raise RuntimeError('Scientific/model scope changed: '+key)
    if (m['clinical_protocol_changed'] or m['automatic_retry'] or m['completed_inference_rerun'] or
            m['prior_partial_rows_pooled'] or m['physical_batch_size'] != 2 or m['extra_attempts_per_oom_job'] != 1):
        raise RuntimeError('Unapproved inference/retry policy')
    if len(m['campaigns']) != 3 or set(m['original_campaigns'].values()) != set(parent['campaigns']):
        raise RuntimeError('Unexpected campaign scope')
    for cp in m['campaigns']:
        old = m['original_campaigns'][cp]
        if m['plan_hashes'][cp] != parent['plan_hashes'][old] or m['campaign_roles'][cp] != parent['campaign_roles'][old]:
            raise RuntimeError('Scientific mapping changed')
    for file, value in m['proofs'].items():
        if digest(read(file)) != value:
            raise RuntimeError('Validation evidence changed')
    for row in m['compiler_adopted']:
        if digest(read(row['result'])) != row['hash']:
            raise RuntimeError('Adopted completion changed')
    for row in m['compiler_history']:
        original = read(row['original_result'])
        if (digest(original) if original else None) != row['original_result_hash']:
            raise RuntimeError('Historical result changed')
    return m


def authorize_job(m, cp, plan, job, allow_launch_log=False):
    if str(cp) not in m['campaigns'] or plan['plan_hash'] != m['plan_hashes'][str(cp)] or job not in plan['jobs']:
        raise RuntimeError('Assignment outside compiler recovery scope')
    rows = [r for r in m['compiler_history'] if r['campaign'] == str(cp) and r['job'] == job['id']]
    if len(rows) != 1 or rows[0]['policy'] not in ('unstarted', 'one_oom_retry'):
        raise RuntimeError('Completed inference cannot run again')
    folder = cp.parent/'jobs'/job['id']
    started = [p for p in folder.glob('attempt[0-9]*')
               if p.is_dir() or not allow_launch_log or p.name != 'attempt1.compiler.log']
    if (folder/'result.json').exists() or started:
        raise RuntimeError('Recovery assignment already started; no automatic retry')
    old = Path(rows[0]['original_result']).parent
    if rows[0]['policy'] == 'one_oom_retry':
        if validate_oom(old, plan, job) != rows[0]['authorization']:
            raise RuntimeError('OOM evidence changed')
    elif (old/'result.json').exists() or list(old.glob('attempt[0-9]*')):
        raise RuntimeError('Previously unstarted assignment changed')
    if job['stage'] != 'test' or not replay.development_passed(m):
        raise RuntimeError('Only formal continuation after all original development audits')


def attempt(path, cp, job, worker, journal, inherited):
    m, plan = load(path), load_plan(cp)
    folder = cp.parent/'jobs'/job['id']
    with FileLease(cp.parent/'locks'/(job['id']+'.lock')) as lease:
        authorize_job(m, cp, plan, job)
        folder.mkdir(parents=True, exist_ok=True); out = folder/'attempt1'
        cmd = [plan['python'], '-u', '-m', 'medtoolbench.supplement.gptoss_compile_runner',
            '--compiler-manifest', str(path), '--plan', str(cp), '--job', job['id'], '--worker', worker['name'], '--out', str(out)]
        journal.persist(folder/'command.json', {'command':cmd,'worker':worker,'manifest_hash':m['manifest_hash']})
        outcome = execute(cmd, folder/'attempt1.compiler.log', worker, path.parent, [*inherited,lease.fileno()], journal)
        result = {'job':job['id'],'model':native.MODEL,'worker':worker['name'],'run':str(out),
            'attempts':[outcome],'manifest_hash':m['manifest_hash'],'inference_reused':False,
            'compiler_recovery_revision':REVISION,'prior_partial_rows_pooled':False}
        proof = read(out/'native_replay_audit.json')
        if outcome['returncode'] == 0 and proof and proof.get('all_pass'):
            result.update(state='complete',episodes=proof['episodes'],
                replay_audit_path=str(out/'native_replay_audit.json'), replay_audit_hash=digest(proof))
            replay.verify_result(result,plan,job,full=True)
        else:
            result.update(state='quarantined',episodes=0,assigned_episodes=len(job['episodes']),
                error=read(out/'native_audit_failure.json') or read(out/'failure.json'),
                reason='Execution or audit failure; no automatic retry')
        journal.persist(folder/'result.json',result)
        return result


def finalize(control,manifest,plans):
    provenance = []
    for cp,plan in plans:
        for job in plan['jobs']:
            result = read(cp.parent/'jobs'/job['id']/'result.json')
            replay.verify_result(result,plan,job,full=True)
            provenance.append({'job':job['id'],'campaign':manifest['campaign_roles'][str(cp)],
                'stage':job['stage'],'episodes':len(job['episodes']),'run':result['run'],
                'runtime':'original' if result.get('compiler_recovery_adopted') else REVISION})
    atomic_json(control/'compute_provenance.json', {'assignments':provenance,
        'same_mathematical_attention':True,'bitwise_generation_equivalence_claimed':False,
        'do_not_pool_wall_time_across_runtime_profiles':True,'partial_failed_rows_pooled':False})
    PARENT_FINALIZE(control,{**manifest,'source_hashes':{**manifest['source_hashes'],
        **manifest['replay_source_hashes'],**manifest['compiler_source_hashes']}},plans)
    atomic_json(control/'compiler_recovery_complete.json', {'all_pass':True,'revision':REVISION,
        'manifest_hash':manifest['manifest_hash'],'reused_formal':5692,'reused_development':260,
        'new_formal':12275,'archived_partial_excluded':80,'entire_project_complete':False,'time':time.time()})


def run(path):
    path = Path(path); m = load(path)
    for raw in m['campaigns']:
        cp = Path(raw); plan = load_plan(cp)
        for job in plan['jobs']:
            result = read(cp.parent/'jobs'/job['id']/'result.json')
            if result:
                replay.verify_result(result,plan,job,full=True)
    for row in m['compiler_history']:
        if row['policy'] == 'one_oom_retry':
            a = row['authorization']
            if evidence_hashes(Path(a['run'])) != a['evidence_sha256'] or checksum(a['log']) != a['log_sha256']:
                raise RuntimeError('Archived OOM evidence changed')
    with FileLease(Path(m['parent_replay_manifest']).parent/'supervisor.lock'):
        replay.stopped(Path(m['parent_replay_manifest']))
        native.load,native.attempt = load,attempt
        native.development_passed,native.finalize = replay.development_passed,finalize
        native.run(path)
