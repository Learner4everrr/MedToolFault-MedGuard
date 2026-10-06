"""Fresh, versioned qualification after a transport bug; no scientific retuning.

Reuse the frozen v1 queue's leases, adoption, retirement and reporting logic.
Only its manifest-loader and per-job launcher hooks are replaced at runtime.
All hook sources are hashed; no frozen v1 file or old successful run is edited.
"""
import argparse
import time
from collections import Counter
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.build import load_gzip
from medtoolbench.research.cluster_execution import FileLease
from . import gptoss_pipeline as base
from .gptoss_protocol import VERSION, decode
from .memory_execution import verify_completion
from .runner import load_plan
from .storage_recovery import atomic_json, execute, read


def source_hashes(root):
    sources = base.source_hashes(root)
    files = [root / 'repo/medtoolbench/supplement' / (name + '.py') for name in
             ('gptoss_protocol', 'gptoss_protocol_model', 'gptoss_protocol_runner', 'gptoss_protocol_pipeline')]
    files += [root / 'repo/scripts' / name for name in
              ('gptoss_protocol_probe.py', 'validate_gptoss_protocol.py', 'run_gptoss_protocol.py')]
    files += [root / 'repo/tests/test_gptoss_protocol.py']
    return {**sources, **{str(p): digest(p.read_text()) for p in files}}


def checked_proof(path, sources):
    proof = read(path)
    if not proof or not proof.get('all_pass') or proof.get('diagnostic_only') or proof.get('held_out_tasks_used') is not False:
        raise RuntimeError('Passing non-diagnostic, non-held-out proof required: ' + str(path))
    for name, checksum in proof['source_hashes'].items():
        if sources.get(name) != checksum:
            raise RuntimeError('Stale proof: ' + name)
    return proof


def prepare(control, roundtrip, unit):
    root = Path(__file__).resolve().parents[3]
    if (control / 'manifest.json').exists():
        raise RuntimeError('Never overwrite a protocol revision')
    old_control = root / 'data/gptoss_replacement_v1'
    old = base.load(old_control / 'manifest.json')
    sources = source_hashes(root)
    probe = checked_proof(roundtrip, sources)
    tests = checked_proof(unit, sources)
    if tests['source_hashes'] != sources:
        raise RuntimeError('Full exact-source regression receipt required')
    if probe.get('synthetic_model_roundtrips') != 4 or probe.get('deterministic_tool_roundtrips') != 7:
        raise RuntimeError('Both real-model protocols and all declared tool dispatchers must be exercised')
    handoff = read(control / 'handoff.json')
    if not handoff or handoff['formal_gptoss_attempts_observed'] != 0:
        raise RuntimeError('Requires explicit zero-formal-attempt handoff')
    proofs = {str(p): digest(read(p)) for p in (roundtrip, unit,
        old_control / 'manifest.json', old_control / 'mask_equivalence_v1/complete.json',
        old_control / 'artifacts.json', Path(old['smoke']), Path(old['unit_proof']))}
    new_paths, hashes = [], {}
    for index, raw in enumerate(old['old_campaigns']):
        parent_path = Path(raw)
        parent = load_plan(parent_path)
        path = control / ('campaign_' + 'abcd'[index]) / 'plan.json'
        if path.exists():
            raise RuntimeError('Partial preparation exists')
        # Reuse exactly the compute-validation inputs and original model matrix.
        plan = base.replacement_plan(parent, parent_path, root, old_control,
            Path(old['smoke']), sources, Path(old['unit_proof']))
        plan.pop('plan_hash')
        plan['session'] = 'medtoolbench-protocol-v2'
        plan['gptoss_protocol_repair'] = {
            'version': VERSION, 'roundtrip_proof': str(roundtrip), 'unit_proof': str(unit),
            'supersedes': str(old_control / 'manifest.json'),
            'all_prior_gptoss_development_excluded': True, 'fresh_full_matrix': True,
            'transport_change': 'Unwrap explicit allowlisted native Harmony tool frames; add external tool selector only when absent',
            'clinical_prompts_labels_budgets_grader_changed': False,
            'compute_template_sampling_and_raw_token_accounting_changed': False,
            'feedback_role_changed': False, 'reasoning_json_executable': False,
            'selection': 'Adapter defect repaired on synthetic fixtures and development traces only; no held-out outcome selection'}
        for file in (roundtrip, unit, control / 'handoff.json'):
            plan['input_hashes'][str(file)] = digest(file.read_bytes().hex())
        plan['plan_hash'] = digest(plan)
        atomic_json(path, plan)
        new_paths.append(str(path))
        hashes[str(parent_path)], hashes[str(path)] = parent['plan_hash'], plan['plan_hash']
    manifest = {'created_at': time.time(), 'project': str(root), 'source_hashes': sources,
        'retired_models': [base.RETIRED], 'target': base.TARGET, 'old_campaigns': old['old_campaigns'],
        'new_campaigns': new_paths, 'plan_hashes': hashes, 'handoff_hash': digest(handoff),
        'proofs': proofs, 'execution_manifest': old['execution_manifest'], 'inference_api': False,
        'max_attempts_per_new_job': 1, 'protocol_version': VERSION,
        'failure_policy': 'Quarantine infrastructure errors; no accuracy-based retries or transport-invalid output reuse',
        'scope': old['scope'], 'old_gptoss_development_disposition': 'Deleted at user request; ownership/invalidation snapshot retained'}
    manifest['manifest_hash'] = digest(manifest)
    atomic_json(control / 'manifest.json', manifest)


def load(path):
    manifest = read(path)
    wanted = manifest.pop('manifest_hash')
    if digest(manifest) != wanted or manifest['retired_models'] != [base.RETIRED] or manifest['protocol_version'] != VERSION:
        raise RuntimeError('Protocol manifest or retirement policy changed')
    manifest['manifest_hash'] = wanted
    for raw, checksum in manifest['source_hashes'].items():
        if digest(Path(raw).read_text()) != checksum:
            raise RuntimeError('Protocol source changed: ' + raw)
    if digest(read(path.parent / 'handoff.json')) != manifest['handoff_hash']:
        raise RuntimeError('Ownership snapshot changed')
    for raw, checksum in manifest['proofs'].items():
        if digest(read(raw)) != checksum:
            raise RuntimeError('Validation proof changed: ' + raw)
    return manifest


def audit_protocol(out, plan):
    mode = 'external' if plan['runner_module'].endswith('external_runner') else 'fhir'
    counts, errors, generations = Counter(), [], 0
    for file in sorted((out / 'episodes').glob('*.json.gz')):
        row = load_gzip(file)
        for index, g in enumerate(row['generations']):
            generations += 1
            text, status, metadata = decode(g['raw_harmony'], mode)
            counts[status] += 1
            if (text, status, metadata) != (g['text'], g['harmony_status'], g.get('protocol_bridge')):
                errors.append({'file': str(file), 'generation': index, 'error': 'raw-stream replay mismatch'})
    proof = {'all_pass': not errors and generations > 0, 'generations': generations,
        'statuses': dict(counts), 'errors': errors, 'clinical_accuracy_gate': False,
        'plan_hash': plan['plan_hash'], 'version': VERSION}
    atomic_json(out / 'protocol_audit.json', proof)
    if not proof['all_pass']:
        raise RuntimeError('Independent transport replay failed')
    return proof


def attempt(path, cp, job, worker, journal, inherited):
    manifest, plan = load(path), load_plan(cp)
    if job['model'] == base.RETIRED:
        raise RuntimeError('Gemma permanently retired')
    folder = cp.parent / 'jobs' / job['id']
    with FileLease(cp.parent / 'locks' / (job['id'] + '.lock')) as lease:
        if (folder / 'result.json').exists() or list(folder.glob('attempt[0-9]*')):
            raise RuntimeError('Never overwrite or retry prior evidence')
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / 'attempt1'
        if job['model'] == base.TARGET:
            cmd = [plan['python'], '-u', '-m', 'medtoolbench.supplement.gptoss_protocol_runner']
        else:
            cmd = [plan['python'], '-u', '-m', 'medtoolbench.supplement.isolated_runner',
                '--execution-manifest', manifest['execution_manifest']]
        cmd += ['--plan', str(cp), '--job', job['id'], '--worker', worker['name'], '--out', str(out)]
        journal.persist(folder / 'protocol_revision_command.json', {'command': cmd, 'worker': worker,
            'manifest_hash': manifest['manifest_hash'], 'scientific_plan_hash': plan['plan_hash']})
        outcome = execute(cmd, folder / 'attempt1.protocol_revision.log', worker, path.parent,
            [*inherited, lease.fileno()], journal)
        result = {'job': job['id'], 'model': job['model'], 'worker': worker['name'], 'run': str(out),
            'attempts': [outcome], 'migration_manifest_hash': manifest['manifest_hash']}
        if outcome['returncode'] == 0 and (out / 'complete.json').exists():
            proof = verify_completion(out, plan, job)
            if job['model'] == base.TARGET:
                result['protocol_audit_hash'] = digest(audit_protocol(out, plan))
            result.update(state='complete', episodes=proof['episodes'])
        else:
            if (out / 'audit.json').exists() or (out / 'complete.json').exists():
                raise RuntimeError('Audit-invalid result; never accept or retry')
            result.update(state='quarantined', episodes=0, error=read(out / 'failure.json'),
                          assigned_episodes=len(job['episodes']))
        journal.persist(folder / 'result.json', result)
        return result


def run(path):
    # Explicit, hashed operational hooks; frozen scheduling code is unchanged.
    base.load = load
    base.attempt = attempt
    base.run(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--control', type=Path, default=Path('data/gptoss_protocol_repair_v2'))
    parser.add_argument('--roundtrip', type=Path)
    parser.add_argument('--unit', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.control.resolve(), args.roundtrip.resolve(), args.unit.resolve())
    else:
        run(args.control.resolve() / 'manifest.json')
