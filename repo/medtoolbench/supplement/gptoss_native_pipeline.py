"""Fresh GPT-OSS A/C/D cohort. Old model evidence is never rerun or pooled.

One exact-source synthetic proof and all 260 fresh development assignments
precede formal dispatch. No success-rate threshold or accuracy-based retry.
"""
import argparse
import copy
import hashlib
import os
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import AllocatedGpuLease, FileLease, validate_worker
from .gptoss_model import REVISION, REASONING
from .gptoss_native import VERSION
from .gptoss_pipeline import source_hashes as compute_sources
from .isolated_execution import model_gate
from .memory_execution import verify_completion
from .runner import load_plan
from .storage_recovery import Journal, atomic_json, execute, read

ROOT = Path(__file__).resolve().parents[3]
CONTROL = ROOT / 'data/gptoss_acd_native_v3'
CAMPAIGNS = {'a': 'supplement_a_v3', 'c': 'supplement_c_v4', 'd': 'supplement_d_v3'}
COUNTS = {'a': (112, 12291), 'c': (84, 4836), 'd': (64, 840)}
MODEL = 'gptoss20b'


def source_hashes():
    files = [Path(__file__).with_name(n + '.py').resolve() for n in
        ('gptoss_native', 'gptoss_native_model', 'gptoss_native_runner', 'gptoss_native_pipeline',
         'concurrent_repair', 'profile_runner', 'profile_store', 'corrected_runner',
         'storage_recovery', 'isolated_execution', 'memory_execution', 'gptoss_protocol')]
    files += [ROOT / 'repo/scripts' / n for n in ('gptoss_native_probe.py', 'run_gptoss_native.py', 'validate_gptoss_native.py')]
    files += [ROOT / 'repo/tests/test_gptoss_native.py']
    return {**compute_sources(ROOT), **{str(f): digest(f.read_text()) for f in files}}


def scientific_job(job):
    return {k: v for k, v in job.items() if k not in ('id', 'model')}


def native_plan(parent, cp, inputs, sources):
    if cp.parent.name not in CAMPAIGNS.values():
        raise RuntimeError('Only A/C/D are authorized')
    original = [j for j in parent['jobs'] if j['model'] == 'qwen35_9b']
    if not original:
        raise RuntimeError('Missing complete matched template')
    plan = copy.deepcopy(parent)
    plan.pop('plan_hash')
    plan['jobs'] = [dict(copy.deepcopy(j), model=MODEL, id='native--' + j['id'].replace('qwen35_9b', MODEL)) for j in original]
    if [scientific_job(j) for j in original] != [scientific_job(j) for j in plan['jobs']]:
        raise RuntimeError('Scientific assignment changed')
    plan['systems'] = [{'name': MODEL, 'repo_id': 'openai/gpt-oss-20b', 'family': 'GPT-OSS',
        'path': str(ROOT / 'data/models/gpt-oss-20b'), 'revision': REVISION, 'thinking': True,
        'reasoning_effort': REASONING, 'batch_size': 2, 'quantization': 'native MXFP4', 'inference_api': False}]
    plan.update(python=str(ROOT / 'env_gptoss/bin/python'), created_at=time.time(),
                session='medtoolbench-gptoss-acd-v3', require_oracle=False)
    plan['source_hashes'].update(sources)
    plan['input_hashes'].update({str(f): digest(f.read_bytes().hex()) for f in inputs})
    plan['native_amendment'] = {'version': VERSION, 'parent_plan': str(cp), 'parent_plan_hash': parent['plan_hash'],
        'parent_oracle': str(cp.parent / 'oracle/complete.json') if parent.get('require_oracle') else None,
        'oracle_reuse': 'Identical tasks, fixtures, hazards and graders; environment oracle is model-independent',
        'serialization_changes': ['Declared native FHIR tool schema', 'Transport-only developer instruction',
                                  'Native tool calls and tool-role feedback with tool reasoning retained'],
        'clinical_base_prompt_preserved': True, 'serialization_token_costs_counted': True,
        'labels_faults_grader_budgets_seeds_unchanged': True, 'reasoning_in_output_budget': True,
        'argument_repair': False, 'old_gptoss_versions_pooled': False, 'clinical_accuracy_gate': False,
        'selection': 'User requested fourth model after viewing three-model supplemental results; exploratory amendment',
        'concurrent_proxy': 'Previously certified rejection-preserving adapter', 'json_envelope_candidate_enabled': False}
    plan['plan_hash'] = digest(plan)
    return plan


def checked_proof(path, sources):
    p = read(path)
    if not p or not p.get('all_pass') or p.get('held_out_tasks_used') is not False or p.get('source_hashes') != sources:
        raise RuntimeError('Missing or stale exact-source proof: ' + str(path))
    return p


def prepare(control, probe, unit):
    if (control / 'manifest.json').exists() or (control / 'campaign_a').exists():
        raise RuntimeError('Never overwrite a native cohort or partial preparation')
    sources = source_hashes()
    proof = checked_proof(probe, sources)
    checked_proof(unit, sources)
    if proof.get('successful_read_write_feedback_conversations') != 2 or not proof.get('negative_calls_preserve_payloads'):
        raise RuntimeError('Actual successful end-to-end synthetic tool use is required')
    prior = ROOT / 'data/acd_llama_expansion_v1/complete.json'
    if not read(prior) or not read(prior).get('all_pass'):
        raise RuntimeError('Prior three-model cohort is not complete')
    old = ROOT / 'data/gptoss_replacement_v1'
    artifact = old / 'artifacts.json'
    receipt = read(artifact)
    if not receipt or not receipt.get('all_pass') or receipt['revision'] != REVISION:
        raise RuntimeError('Pinned model artifact receipt required')
    metadata = {}
    for name, r in receipt['files'].items():
        file = ROOT / 'data/models/gpt-oss-20b' / name
        if file.stat().st_size != r['size']:
            raise RuntimeError('Pinned model file size changed')
        if not name.endswith('.safetensors'):
            checksum = hashlib.sha256(file.read_bytes()).hexdigest()
            if checksum != r['sha256']:
                raise RuntimeError('Pinned model metadata changed')
            metadata[name] = checksum
    artifact_now = control / 'artifacts_current.json'
    atomic_json(artifact_now, {'all_pass': True, 'original_sha256_receipt': str(artifact),
        'metadata_sha256': metadata, 'all_sizes_checked': True, 'weight_sha256_recomputed': False,
        'time': time.time(), 'inference_api': False})
    smoke, mask = old / 'smoke_v5/complete.json', old / 'mask_equivalence_v1/complete.json'
    if not read(smoke).get('all_pass') or not read(mask).get('all_pass'):
        raise RuntimeError('Validated full-budget compute required')
    if read(smoke)['adapter_hash'] != sources[str(Path(__file__).with_name('gptoss_model.py'))]:
        raise RuntimeError('Compute validation source mismatch')
    if read(mask)['repair_hash'] != sources[str(Path(__file__).with_name('gptoss_mask.py'))]:
        raise RuntimeError('Mask validation source mismatch')
    repair = ROOT / 'data/concurrent_repair_validation_v2/complete.json'
    if not read(repair).get('all_pass'):
        raise RuntimeError('Concurrent transport validation missing')
    common = [probe, unit, prior, artifact, artifact_now, smoke, mask, repair]
    campaigns, hashes, roles = [], {}, {}
    for letter, name in CAMPAIGNS.items():
        cp = ROOT / 'data' / name / 'plan.json'
        parent = load_plan(cp)
        inputs = [cp, *common]
        if parent.get('require_oracle'):
            oracle = cp.parent / 'oracle/complete.json'
            op = read(oracle)
            if not op or not op.get('all_pass') or op['plan_hash'] != parent['plan_hash']:
                raise RuntimeError('Parent environment oracle invalid')
            inputs.append(oracle)
        plan = native_plan(parent, cp, inputs, sources)
        totals = tuple(sum(len(j['episodes']) for j in plan['jobs'] if j['stage'] == stage) for stage in ('qualification', 'test'))
        if totals != COUNTS[letter]:
            raise RuntimeError('Incomplete native matrix')
        dest = control / ('campaign_' + letter) / 'plan.json'
        atomic_json(dest, plan)
        campaigns.append(str(dest)); hashes[str(dest)] = plan['plan_hash']; roles[str(dest)] = letter.upper()
    m = {'version': VERSION, 'source_hashes': sources, 'campaigns': campaigns, 'plan_hashes': hashes,
        'campaign_roles': roles, 'model': MODEL, 'created_at': time.time(), 'inference_api': False,
        'proofs': {str(f): digest(read(f)) for f in common}, 'all_development_before_formal': True,
        'formal_episodes': 17967, 'development_episodes': 260, 'max_attempts_per_job': 1,
        'prior_three_model_complete': str(prior), 'automatic_b_resume': False, 'gemma_retired': True,
        'old_gptoss_pooled': False, 'clinical_accuracy_gate': False}
    m['manifest_hash'] = digest(m)
    atomic_json(control / 'manifest.json', m)
    load(control / 'manifest.json')


def load(path):
    m = read(path)
    if not m:
        raise RuntimeError('Native manifest missing')
    wanted = m.pop('manifest_hash')
    if digest(m) != wanted or m['source_hashes'] != source_hashes():
        raise RuntimeError('Native manifest/source changed')
    m['manifest_hash'] = wanted
    if m['version'] != VERSION or m['model'] != MODEL or m['automatic_b_resume'] or m['old_gptoss_pooled']:
        raise RuntimeError('Native scope changed')
    if (not m['all_development_before_formal'] or m['max_attempts_per_job'] != 1 or
            m['inference_api'] or not m['gemma_retired'] or m['clinical_accuracy_gate'] or
            (m['development_episodes'], m['formal_episodes']) != (260, 17967)):
        raise RuntimeError('Native execution policy changed')
    if len(m['campaigns']) != 3 or set(m['campaign_roles'].values()) != {'A', 'C', 'D'}:
        raise RuntimeError('Only complete A/C/D scope is allowed')
    for name, checksum in m['proofs'].items():
        if digest(read(name)) != checksum:
            raise RuntimeError('Native proof changed: ' + name)
    return m


def development_passed(manifest):
    total = 0
    for raw in manifest['campaigns']:
        cp = Path(raw); plan = load_plan(cp)
        if model_gate(cp, plan, MODEL) != 'passed':
            return False
        for j in plan['jobs']:
            if j['stage'] != 'qualification':
                continue
            r = read(cp.parent / 'jobs' / j['id'] / 'result.json')
            proof = read(Path(r['run']) / 'native_audit.json')
            if not proof or not proof.get('all_pass') or proof.get('version') != VERSION or digest(proof) != r.get('native_audit_hash'):
                return False
            total += len(j['episodes'])
    return total == 260


def candidates(plans, active, dev_passed):
    ready = []
    for index, (cp, plan) in enumerate(plans):
        for j in plan['jobs']:
            if j['model'] != MODEL:
                raise RuntimeError('Unexpected model')
            folder = cp.parent / 'jobs' / j['id']
            if (str(cp), j['id']) in active or (folder / 'result.json').exists() or list(folder.glob('attempt[0-9]*')):
                continue
            if j['stage'] == 'test' and not dev_passed:
                continue
            ready.append((0 if j['stage'] == 'qualification' else 1, index, cp, plan, j))
    return ready


def attempt(path, cp, job, worker, journal, inherited):
    m, plan = load(path), load_plan(cp)
    folder = cp.parent / 'jobs' / job['id']
    with FileLease(cp.parent / 'locks' / (job['id'] + '.lock')) as lease:
        if (folder / 'result.json').exists() or list(folder.glob('attempt[0-9]*')):
            raise RuntimeError('Never overwrite or blindly retry evidence')
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / 'attempt1'
        cmd = [plan['python'], '-u', '-m', 'medtoolbench.supplement.gptoss_native_runner',
            '--native-manifest', str(path), '--plan', str(cp), '--job', job['id'], '--worker', worker['name'], '--out', str(out)]
        journal.persist(folder / 'command.json', {'command': cmd, 'worker': worker, 'manifest_hash': m['manifest_hash']})
        outcome = execute(cmd, folder / 'attempt1.native.log', worker, path.parent, [*inherited, lease.fileno()], journal)
        result = {'job': job['id'], 'model': MODEL, 'worker': worker['name'], 'run': str(out),
            'attempts': [outcome], 'manifest_hash': m['manifest_hash']}
        np = read(out / 'native_audit.json')
        if (outcome['returncode'] == 0 and np and np.get('all_pass') and
                np.get('version') == VERSION and np.get('plan_hash') == plan['plan_hash']):
            proof = verify_completion(out, plan, job)
            result.update(state='complete', episodes=proof['episodes'], native_audit_hash=digest(np))
        else:
            result.update(state='quarantined', episodes=0, assigned_episodes=len(job['episodes']),
                error=read(out / 'failure.json'), reason='Runner or clinical/native audit incomplete; no automatic retry')
        journal.persist(folder / 'result.json', result)
        return result


def verify_reference(result, plan, job, cache=None):
    """Resolve an analysis-only reference back to its actual frozen run plan."""
    if not result.get('analysis_reference_only'):
        return verify_completion(Path(result['run']), plan, job)
    source = Path(result['source_result'])
    original = read(source)
    if not original or digest(original) != result['source_result_hash'] or original['run'] != result['run']:
        raise RuntimeError('Analysis reference changed')
    paths = [Path(p) for p, h in plan['source_plans'].items() if h == result['source_plan_hash']]
    if len(paths) != 1:
        raise RuntimeError('Ambiguous analysis source plan')
    cp = paths[0]
    cache = {} if cache is None else cache
    if str(cp) not in cache:
        cache[str(cp)] = load_plan(cp)
    parent = cache[str(cp)]
    if parent['plan_hash'] != result['source_plan_hash']:
        raise RuntimeError('Analysis source plan changed')
    original_job = next(j for j in parent['jobs'] if j['id'] == original['job'])
    if scientific_job(original_job) != scientific_job(job) or original_job['model'] != job['model']:
        raise RuntimeError('Analysis assignment changed')
    return verify_reference(original, parent, original_job, cache)


def finalize(control, manifest, plans):
    from .report import generate
    reference_cache = {}
    for cp, plan in plans:
        generate(cp)
        letter = manifest['campaign_roles'][str(cp)].lower()
        oldcp = ROOT / 'data/acd_llama_expansion_v1' / ('combined_' + letter) / 'plan.json'
        old = load_plan(oldcp)
        view = copy.deepcopy(old); view.pop('plan_hash')
        view['jobs'] += copy.deepcopy(plan['jobs']); view['systems'] += copy.deepcopy(plan['systems'])
        if len({j['id'] for j in view['jobs']}) != len(view['jobs']):
            raise RuntimeError('Overlapping analysis identifiers')
        view.update(analysis_only=True, native_cohort_separate=True,
            full_matrix_models=['qwen35_9b', 'ministral3_8b', 'llama31_8b', MODEL])
        view['source_hashes'].update(manifest['source_hashes'])
        view['input_hashes'].update(plan['input_hashes'])
        view['source_plans'] = {str(oldcp): old['plan_hash'], str(cp): plan['plan_hash']}
        view['plan_hash'] = digest(view)
        vp = control / ('combined_' + letter) / 'plan.json'
        if vp.exists():
            raise RuntimeError('Partial combined report needs explicit reconciliation')
        atomic_json(vp, view)
        for source_cp, source_plan in ((oldcp, old), (cp, plan)):
            for j in source_plan['jobs']:
                source = source_cp.parent / 'jobs' / j['id'] / 'result.json'
                r = read(source)
                if not r or r.get('state') != 'complete':
                    raise RuntimeError('Incomplete report reference')
                verify_reference(r, source_plan, j, reference_cache)
                atomic_json(vp.parent / 'jobs' / j['id'] / 'result.json', {**r, 'analysis_reference_only': True,
                    'source_result': str(source), 'source_result_hash': digest(r), 'source_plan_hash': source_plan['plan_hash']})
        generate(vp)
    atomic_json(control / 'complete.json', {'all_pass': True, 'manifest_hash': manifest['manifest_hash'],
        'time': time.time(), 'new_formal_episodes': 17967, 'new_development_episodes': 260,
        'scope': 'Four-model ACD; GPT-OSS native transport amendment explicitly separate', 'entire_project_complete': False})


def run(path):
    m = load(path); control = path.parent
    plans = [(Path(p), load_plan(Path(p))) for p in m['campaigns']]
    if any(p['plan_hash'] != m['plan_hashes'][str(cp)] for cp, p in plans):
        raise RuntimeError('Native plan identity mismatch')
    workers = plans[0][1]['workers']
    for w in workers:
        validate_worker(plans[0][1], w['name'])
    journal = Journal(control, Path(tempfile.mkdtemp(prefix='medtoolbench-gptoss-native-')))
    busy, fatal, cursor, dev_ok = {}, None, 0, False
    with FileLease(control / 'supervisor.lock') as owner, ThreadPoolExecutor(max_workers=len(workers)) as pool:
        if (control / 'started.json').exists():
            raise RuntimeError('Restart requires explicit ownership reconciliation')
        journal.persist(control / 'started.json', {'pid': os.getpid(), 'time': time.time(), 'manifest_hash': m['manifest_hash']})
        while True:
            try:
                load(path)
                for name, (future, _) in list(busy.items()):
                    if future.done():
                        del busy[name]
                        result = future.result()
                        if result['state'] != 'complete':
                            fatal = 'Quarantined native job: ' + result['job']
                pending, counts = [], {'qualification': 0, 'test': 0}
                for cp, plan in plans:
                    for j in plan['jobs']:
                        r = read(cp.parent / 'jobs' / j['id'] / 'result.json')
                        if r and r.get('state') == 'complete':
                            counts[j['stage']] += len(j['episodes'])
                        else:
                            pending.append({'plan': str(cp), 'job': j['id'], 'stage': j['stage'], 'episodes': len(j['episodes'])})
                            if r:
                                fatal = 'Quarantined native job: ' + j['id']
                if not dev_ok and counts['qualification'] == 260:
                    dev_ok = development_passed(m)
                    if not dev_ok:
                        raise RuntimeError('Development audit inconsistency')
                    journal.persist(control / 'development_complete.json', {'all_pass': True, 'episodes': 260,
                        'clinical_accuracy_gate': False, 'manifest_hash': m['manifest_hash'], 'time': time.time()})
                if not fatal and journal.healthy():
                    active = {key for _, key in busy.values()}
                    for w in workers:
                        if w['name'] in busy:
                            continue
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(w['gpu'])
                        try:
                            with AllocatedGpuLease(w):
                                pass
                        except BlockingIOError:
                            continue
                        ready = candidates(plans, active, dev_ok)
                        if not ready:
                            break
                        ready.sort(key=lambda r: (r[0], (r[1] - cursor) % len(plans), r[4]['id']))
                        _, index, cp, plan, j = ready[0]; cursor = (index + 1) % len(plans)
                        key = (str(cp), j['id'])
                        busy[w['name']] = (pool.submit(attempt, path, cp, j, w, journal, [owner.fileno()]), key)
                        active.add(key)
                journal.put(control / 'pending.json', {'jobs': pending, 'time': time.time()})
                journal.put(control / 'status.json', {'state': 'integrity_stop' if fatal else 'formal' if dev_ok else 'development',
                    'pid': os.getpid(), 'busy': {k: list(v[1]) for k, v in busy.items()}, 'counts': counts,
                    'heartbeat': time.time(), 'error': fatal, 'manifest_hash': m['manifest_hash']})
                if not busy and (fatal or not pending):
                    if fatal:
                        journal.persist(control / 'needs_review.json', {'error': fatal, 'pending_jobs': len(pending), 'time': time.time()})
                    else:
                        journal.persist(control / 'inference_complete.json', {'all_pass': True, 'time': time.time()})
                        finalize(control, m, plans)
                        journal.persist(control / 'status.json', {'state': 'complete', 'counts': counts, 'heartbeat': time.time()})
                    return
            except Exception:
                fatal = traceback.format_exc()
                journal.put(control / 'integrity_failure.json', {'error': fatal, 'time': time.time()})
                if not busy:
                    journal.persist(control / 'needs_review.json', {'error': fatal, 'time': time.time()})
                    return
            time.sleep(10)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'run'])
    p.add_argument('--control', type=Path, default=CONTROL); p.add_argument('--probe', type=Path); p.add_argument('--unit', type=Path)
    a = p.parse_args()
    if a.command == 'prepare':
        prepare(a.control.resolve(), a.probe.resolve(), a.unit.resolve())
    else:
        run(a.control.resolve() / 'manifest.json')
