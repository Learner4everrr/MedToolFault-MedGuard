"""Other-model-only continuation; never auto-resume deferred GPT-OSS/Gemma.

Reuse the frozen eight-lane supervisor, physical/job leases, quota journal,
qualification gates and independently audited child adoption. Only operational
hooks change. Original scientific plans, inference code and results are intact.
"""
import argparse
import time
from pathlib import Path

from medtoolbench.storage import digest
from . import gptoss_pipeline as engine
from .memory_execution import verify_completion
from .runner import load_plan
from .storage_recovery import atomic_json, read

ALLOWED = frozenset(('qwen35_9b', 'ministral3_8b', 'llama31_8b', 'phi4_mini'))
DEFERRED = ('gptoss20b', 'gemma3_12b')
_ATTEMPT = engine.attempt
_ELIGIBLE = engine.eligible_jobs


def source_hashes(root):
    paths = [Path(__file__).resolve(), Path(engine.__file__).resolve(),
        root / 'repo/tests/test_non_gptoss_priority.py',
        root / 'repo/scripts/validate_non_gptoss_priority.py',
        root / 'repo/scripts/run_non_gptoss_priority.py']
    return {str(p): digest(p.read_text()) for p in paths}


def require_allowed(model):
    if model not in ALLOWED:
        raise RuntimeError('Deferred or undeclared model cannot be dispatched: ' + model)


def prepare(control, unit):
    root = Path(__file__).resolve().parents[3]
    if (control / 'manifest.json').exists():
        raise RuntimeError('Never overwrite a priority continuation')
    sources = source_hashes(root)
    proof = read(unit)
    if not proof or not proof.get('all_pass') or proof.get('source_hashes') != sources:
        raise RuntimeError('Exact-source passing CPU validation required')
    prior_path = root / 'data/gptoss_protocol_repair_v2/manifest.json'
    from .gptoss_protocol_pipeline import load as load_prior
    prior = load_prior(prior_path)
    handoff_path = root / 'data/gptoss_native_v3/handoff.json'
    handoff = read(handoff_path)
    if not handoff or engine.process_cmdline(handoff['old_controller_pid']):
        raise RuntimeError('Previous dispatcher must have exited')
    for row in handoff['gptoss_children_stopped']:
        if engine.process_cmdline(row['pid']):
            raise RuntimeError('Deferred GPT-OSS worker still exists')
    for row in handoff['live_children_preserved']:
        require_allowed(row['config']['job']['model'])
    hashes, inventory = {}, []
    for raw in prior['old_campaigns']:
        cp = Path(raw)
        plan = load_plan(cp)
        if any(j['model'] not in ALLOWED | {'gemma3_12b'} for j in plan['jobs']):
            raise RuntimeError('Original plans contain an unexpected model')
        hashes[str(cp)] = plan['plan_hash']
        for job in plan['jobs']:
            if job['model'] not in ALLOWED:
                continue
            folder = cp.parent / 'jobs' / job['id']
            result = read(folder / 'result.json')
            if result and result.get('state') == 'complete':
                verify_completion(Path(result['run']), plan, job)
            inventory.append({'plan': str(cp), 'job': job['id'], 'model': job['model'],
                'stage': job['stage'], 'episodes': len(job['episodes']),
                'state': result.get('state') if result else 'started' if list(folder.glob('attempt[0-9]*')) else 'pending'})
    atomic_json(control / 'handoff.json', {**handoff,
        'priority_instruction': 'Finish other models first; GPT-OSS reconsideration deferred, never automatic',
        'source_snapshot': str(handoff_path)})
    atomic_json(control / 'initial_inventory.json', {'time': time.time(), 'jobs': inventory})
    manifest = {'created_at': time.time(), 'project': str(root), 'source_hashes': sources,
        'allowed_models': sorted(ALLOWED), 'deferred_models': list(DEFERRED),
        'retired_models': ['gemma3_12b'], 'target': None,
        'old_campaigns': prior['old_campaigns'], 'new_campaigns': [], 'plan_hashes': hashes,
        'execution_manifest': prior['execution_manifest'],
        'handoff_hash': digest(read(control / 'handoff.json')),
        'input_hashes': {str(p): digest(read(p)) for p in (unit, prior_path, handoff_path)},
        'inference_api': False, 'original_scientific_plans_unchanged': True,
        'no_accuracy_based_retry': True, 'automatic_deferred_resume': False,
        'failure_policy': 'Preserve failed jobs and quarantine affected model; other healthy models continue. No blind retries.'}
    manifest['manifest_hash'] = digest(manifest)
    atomic_json(control / 'manifest.json', manifest)


def load(path):
    manifest = read(path)
    wanted = manifest.pop('manifest_hash')
    if digest(manifest) != wanted:
        raise RuntimeError('Priority manifest changed')
    manifest['manifest_hash'] = wanted
    if (manifest['allowed_models'] != sorted(ALLOWED) or manifest['deferred_models'] != list(DEFERRED)
            or manifest['new_campaigns'] or manifest['automatic_deferred_resume']):
        raise RuntimeError('Other-model-only dispatch policy changed')
    for raw, checksum in manifest['source_hashes'].items():
        if digest(Path(raw).read_text()) != checksum:
            raise RuntimeError('Priority source changed: ' + raw)
    if digest(read(path.parent / 'handoff.json')) != manifest['handoff_hash']:
        raise RuntimeError('Ownership handoff changed')
    for raw, checksum in manifest['input_hashes'].items():
        if digest(read(raw)) != checksum:
            raise RuntimeError('Priority input changed: ' + raw)
    return manifest


def eligible(plans, active, blocked):
    return [row for row in _ELIGIBLE(plans, active, blocked)
            if row[4]['model'] in ALLOWED]


def attempt(path, cp, job, worker, journal, inherited):
    require_allowed(job['model'])
    load(path)
    return _ATTEMPT(path, cp, job, worker, journal, inherited)


def finalize(path, manifest, plans):
    """Completion of this bounded cohort is not completion of the whole project."""
    rows = []
    for cp, plan in plans:
        for job in plan['jobs']:
            if job['model'] not in ALLOWED:
                continue
            result = read(cp.parent / 'jobs' / job['id'] / 'result.json')
            if not result or result.get('state') != 'complete':
                raise RuntimeError('Other-model cohort still has unresolved jobs')
            proof = verify_completion(Path(result['run']), plan, job)
            rows.append({'plan': str(cp), 'job': job['id'], 'model': job['model'],
                'stage': job['stage'], 'episodes': proof['episodes'], 'run': result['run'],
                'result_hash': digest(result)})
    atomic_json(path.parent / 'complete.json', {'all_pass': True, 'time': time.time(),
        'manifest_hash': manifest['manifest_hash'], 'scope': 'Allowed other-model supplemental cohort only',
        'formal_episodes': sum(r['episodes'] for r in rows if r['stage'] == 'test'),
        'development_episodes': sum(r['episodes'] for r in rows if r['stage'] == 'qualification'),
        'jobs': rows, 'deferred_models': list(DEFERRED), 'automatic_deferred_resume': False,
        'entire_project_complete': False})


def run(path):
    engine.load = load
    engine.eligible_jobs = eligible
    engine.attempt = attempt
    engine.finalize = finalize
    engine.run(path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'run'])
    p.add_argument('--control', type=Path, default=Path('data/non_gptoss_priority_v1'))
    p.add_argument('--unit', type=Path)
    a = p.parse_args()
    if a.command == 'prepare':
        prepare(a.control.resolve(), a.unit.resolve())
    else:
        run(a.control.resolve() / 'manifest.json')
