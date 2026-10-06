"""ACD-only priority handoff; let existing B shards drain, never dispatch new B.

This operational amendment does not add models or modify scientific plans.
The separate four/seven-model coverage decision remains explicit.
"""
import argparse
import os
import signal
import time
from pathlib import Path

from medtoolbench.storage import digest
from . import concurrent_pipeline as parent
from . import gptoss_pipeline as engine
from .memory_execution import verify_completion
from .runner import load_plan
from .storage_recovery import atomic_json, read

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / 'data/non_gptoss_priority_v3'
CONTROL = ROOT / 'data/acd_priority_order_v1'
_LOAD = parent.load
_FOLDER = parent.folder_for
_INVENTORY = parent.inventory
_ELIGIBLE = parent.eligible


def is_acd(cp):
    return Path(cp).parent.name in {'supplement_a_v3', 'supplement_c_v4', 'supplement_d_v3'}


def source_hashes():
    files = [Path(__file__).resolve(), ROOT / 'repo/tests/test_acd_order.py']
    return {str(p): digest(p.read_text()) for p in files}


def load(path):
    path = Path(path)
    m = read(path)
    wanted = m.pop('manifest_hash')
    if digest(m) != wanted:
        raise RuntimeError('ACD scheduling manifest changed')
    m['manifest_hash'] = wanted
    base = _LOAD(Path(m['scheduling_parent']))
    if base['manifest_hash'] != m['scheduling_parent_hash']:
        raise RuntimeError('Scheduling parent changed')
    for key in ('old_campaigns', 'plan_hashes', 'allowed_models', 'execution_manifest', 'repair_jobs'):
        if m[key] != base[key]:
            raise RuntimeError('Operational amendment changed scientific scope: ' + key)
    if (m['dispatch_campaigns'] != ['supplement_a_v3', 'supplement_c_v4', 'supplement_d_v3']
            or m['automatic_b_resume'] or m['new_models_added']):
        raise RuntimeError('Undeclared scheduling expansion')
    if m['scheduling_source_hashes'] != source_hashes():
        raise RuntimeError('Scheduling source changed')
    if digest(read(m['scheduling_validation'])) != m['scheduling_validation_hash']:
        raise RuntimeError('CPU validation changed')
    if digest(read(path.parent / 'handoff.json')) != m['handoff_hash']:
        raise RuntimeError('ACD ownership handoff changed')
    return m


def folder_for(control, cp, job, manifest):
    # Certified repair artifacts remain in their ORIGINAL overlay directory.
    return _FOLDER(Path(manifest['scheduling_parent']).parent, cp, job, manifest)


def inventory(control, plans, manifest):
    return _INVENTORY(control, [(cp, p) for cp, p in plans if is_acd(cp)], manifest)


def eligible(control, plans, manifest, active, blocked):
    # Filter plans BEFORE gate evaluation: an old B failure never blocks ACD.
    return _ELIGIBLE(control, [(cp, p) for cp, p in plans if is_acd(cp)], manifest, active, blocked)


def finalize(control, plans, manifest):
    results = []
    deferred = []
    for cp, plan in plans:
        for job in plan['jobs']:
            if job['model'] not in manifest['allowed_models']:
                continue
            file = folder_for(control, cp, job, manifest) / 'result.json'
            r = read(file)
            if not is_acd(cp):
                if not r or r.get('state') != 'complete':
                    deferred.append({'plan': str(cp), 'job': job['id'], 'episodes': len(job['episodes'])})
                continue
            if not r or r.get('state') != 'complete':
                raise RuntimeError('Unresolved ACD assignment')
            proof = verify_completion(Path(r['run']), plan, job)
            results.append({'plan': str(cp), 'job': job['id'], 'model': job['model'],
                'stage': job['stage'], 'episodes': proof['episodes'], 'result': str(file), 'result_hash': digest(r)})
    atomic_json(control / 'complete.json', {'all_pass': True, 'time': time.time(),
        'manifest_hash': manifest['manifest_hash'], 'scope': 'Existing ACD assignments only; no model expansion',
        'jobs': results, 'deferred_b': deferred, 'entire_project_complete': False,
        'four_model_acd_complete': False, 'automatic_b_resume': False})


def handoff(control, validation):
    """Freeze only the verified dispatcher; its independent children keep running."""
    if (control / 'handoff.json').exists() or (control / 'manifest.json').exists():
        raise RuntimeError('Never overwrite a handoff')
    proof = read(validation)
    if not proof or not proof.get('all_pass') or proof.get('source_hashes') != source_hashes():
        raise RuntimeError('Exact-source full CPU regression required')
    base = _LOAD(PRIOR / 'manifest.json')
    state = read(PRIOR / 'status.json')
    pid = state['pid']
    command = engine.process_cmdline(pid)
    if '-m medtoolbench.supplement.concurrent_pipeline run' not in command:
        raise RuntimeError('Unexpected current dispatcher identity')
    os.kill(pid, signal.SIGSTOP)
    try:
        state = read(PRIOR / 'status.json')
        children = []
        if state['preserved_children']:
            raise RuntimeError('Unexpected earlier handoff children; reconcile first')
        for name, (cp, job_id) in state['busy'].items():
            row = read(PRIOR / 'workers' / (name + '.json'))
            config = read(Path(row['out']) / 'config.json')
            plan = load_plan(Path(cp))
            if config['job']['id'] != job_id or config['plan_hash'] != plan['plan_hash']:
                raise RuntimeError('Launch snapshot was changing; retry after resuming')
            current = engine.process_cmdline(row['pid'])
            if current and (row['out'] not in current or '-m medtoolbench.supplement.isolated_runner ' not in current):
                raise RuntimeError('Child identity changed')
            children.append({**row, 'config': config, 'observed_cmdline': current})
        snapshot = {'time': time.time(), 'old_controller_pid': pid, 'old_controller_cmdline': command,
            'old_status': state, 'live_children_preserved': children, 'no_model_children_signalled': True,
            'b_policy': 'Existing shards drain; no new B assignments; no partial rows discarded'}
        manifest = dict(base)
        manifest.pop('manifest_hash')
        manifest.update(scheduling_parent=str(PRIOR / 'manifest.json'),
            scheduling_parent_hash=base['manifest_hash'], scheduling_source_hashes=source_hashes(),
            scheduling_validation=str(validation), scheduling_validation_hash=digest(proof),
            handoff_hash=digest(snapshot), dispatch_campaigns=['supplement_a_v3', 'supplement_c_v4', 'supplement_d_v3'],
            automatic_b_resume=False, new_models_added=[], created_at=time.time())
        manifest['manifest_hash'] = digest(manifest)
        atomic_json(control / 'handoff.json', snapshot)
        atomic_json(control / 'manifest.json', manifest)
        load(control / 'manifest.json')
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
    atomic_json(PRIOR / 'superseded_acd_priority.json', {'time': time.time(),
        'successor': str(control), 'model_children_preserved': True})


def run(path):
    parent.load = load
    parent.folder_for = folder_for
    parent.inventory = inventory
    parent.eligible = eligible
    parent.finalize = finalize
    parent.run(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['handoff', 'run'])
    parser.add_argument('--control', type=Path, default=CONTROL)
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    if args.command == 'handoff':
        handoff(args.control.resolve(), args.validation.resolve())
    else:
        run(args.control.resolve() / 'manifest.json')
