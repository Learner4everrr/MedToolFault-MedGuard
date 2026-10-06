"""CPU and real local HAPI regressions using synthetic inputs, not held-out labels."""
import argparse
import copy
import json
import time
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.storage import digest
from medtoolbench.research.build import save_gzip
from medtoolbench.research.contracts import service, state_payload
from medtoolbench.research.fhir import HapiProcess, normalized_state
from medtoolbench.supplement import runner
from medtoolbench.supplement.concurrent_repair import ConcurrentRejectionProxy
from medtoolbench.supplement.corrected_runner import InterfaceConsistentProxy
from medtoolbench.supplement.input_guard import parse_action
from medtoolbench.supplement.storage_recovery import atomic_json
from medtoolbench.supplement.transport import CONDITIONS, CallBudget, FactorClient

ROOT = Path(__file__).resolve().parents[2]


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT / 'repo/tests/test_concurrent_repair.py']
    paths += [ROOT / 'repo/medtoolbench/supplement' / (n + '.py')
              for n in ('concurrent_repair', 'concurrent_runner')]
    return {str(p): digest(p.read_text()) for p in paths}


def state_without_ids(store):
    return sorted((state_payload(r) for r in store.snapshot()), key=lambda r: json.dumps(r, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root = a.out.resolve()
    if root.exists():
        raise RuntimeError('Never overwrite validation')
    root.mkdir(parents=True)
    sources = source_hashes()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'repo/tests'))
    with (root / 'unit.log').open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or source_hashes() != sources:
        raise RuntimeError('Exact-source full regression required')
    atomic_json(root / 'unit.json', {'all_pass': True, 'tests': result.testsRun,
        'source_hashes': sources, 'time': time.time(), 'held_out_used': False})
    print(json.dumps({'unit_tests': result.testsRun, 'all_pass': True}), flush=True)
    checks, equivalence, old_reproductions = [], [], []
    fixture = [{'resourceType': 'Patient', 'id': 'concurrent-synthetic'}]
    fixture_path = root / 'synthetic-fixture.json.gz'
    save_gzip(fixture_path, fixture)
    body = service('concurrent-synthetic', '4548-4')
    task = {'id': 'synthetic-concurrent', 'patient': 'concurrent-synthetic',
        'hazards': ['clean', 'commit_timeout', 'no_commit_timeout'], 'fixture_hash': digest(fixture),
        'supplement_fixture': str(fixture_path), 'group': 'synthetic', 'family': 'synthetic',
        'template': 0, 'split': 'dev', 'prompt': 'Synthetic transport regression; create the specified service.',
        'contract': {'required_creations': [body], 'answer': {'completed': True}}}
    malformed = {'encoded_date': {**body, 'authoredOn': '2023-11-13T10%3A15%3A00%2B00%3A00'},
                 'resource_mismatch': {**body, 'resourceType': 'MedicationRequest'}}
    settings = {'stress': {'concurrent': True}, 'raw_call_budget': 48}
    try:
        image = ROOT / 'data/upstream/medagentbench.sif'
        with HapiProcess(image, root / 'hapi', heap='2g') as store:
            for name, bad in malformed.items():
                store.reload(fixture)
                old = InterfaceConsistentProxy(store, task, 'commit_timeout', {'concurrent': True})
                try:
                    old.call('POST', '/ServiceRequest', bad)
                except RuntimeError as error:
                    if str(error) != 'Controlled concurrent write failed':
                        raise
                    old_reproductions.append(name)
                else:
                    raise RuntimeError('Synthetic old crash not reproduced')
            for policy in CONDITIONS:
                for hazard in ('commit_timeout', 'no_commit_timeout'):
                    signatures = []
                    for cls in (InterfaceConsistentProxy, ConcurrentRejectionProxy):
                        store.reload(fixture)
                        proxy = cls(store, task, hazard, {'concurrent': True})
                        budget = CallBudget(proxy.call, 48)
                        response = FactorClient(budget, policy).call('POST', '/ServiceRequest', body)
                        signatures.append({'response_status': response['status'], 'state': state_without_ids(store),
                            'triggered': proxy.triggered, 'counts': proxy.counts,
                            'external_count': len(proxy.external_ids), 'tool_calls': budget.used,
                            'trace_statuses': [r['response']['status'] for r in proxy.trace],
                            'mutation_indexes': [r['trace_index'] for r in proxy.mutation_states]})
                    if signatures[0] != signatures[1]:
                        raise RuntimeError('Valid-path semantics differ')
                    equivalence.append({'policy': policy, 'hazard': hazard, 'all_pass': True})
                    for name, bad in malformed.items():
                        out = root / 'synthetic_audits' / (policy + '--' + hazard + '--' + name)
                        out.mkdir(parents=True)
                        (out / 'states').mkdir()
                        (out / 'episodes').mkdir()
                        plan = {'tasks': [task]}
                        key = task['id'] + '--' + hazard + '--' + policy
                        job = {'id': out.name, 'settings': settings,
                               'episodes': [{'key': key, 'task': task['id'], 'hazard': hazard, 'condition': policy}]}
                        plan['plan_hash'] = digest({'synthetic': True, 'task': task, 'job': job})
                        atomic_json(out / 'config.json', {'plan_hash': plan['plan_hash'], 'job': job})
                        with patch.object(runner, 'StressProxy', ConcurrentRejectionProxy), patch(
                                'medtoolbench.research.run.parse_action', parse_action):
                            episode = runner.SupplementEpisode(store, task, hazard, policy, plan, settings)
                            before = normalized_state(store.snapshot())
                            def advance(action):
                                episode.advance({'text': json.dumps(action), 'input_tokens': 1, 'output_tokens': 1,
                                    'batch_seconds_per_item': 0, 'generation_limit': False})
                            advance({'method': 'POST', 'path': '/ServiceRequest', 'body': bad})
                            response = json.loads(episode.messages[-1]['content'])
                            if not (400 <= response['status'] < 500 and normalized_state(store.snapshot()) == before
                                    and not episode.proxy.concurrent_done and not episode.proxy.triggered
                                    and not episode.proxy.mutation_states and not episode.proxy.external_ids
                                    and episode.budget.used == len(episode.proxy.trace)
                                    and episode.proxy.trace[-1]['body'] == bad):
                                raise RuntimeError('Rejected-request semantics failed')
                            advance({'method': 'POST', 'path': '/ServiceRequest', 'body': body})
                            if len(episode.proxy.external_ids) != 1 or episode.proxy.triggered != 1:
                                raise RuntimeError('Subsequent valid fault did not execute')
                            advance({'final': {'completed': True}})
                            row = episode.result()
                        initial = row.pop('before')
                        ref = 'states/' + digest(initial) + '.json.gz'
                        save_gzip(out / ref, initial)
                        row.update(before_ref=ref, total_tokens=row['input_tokens'] + row['output_tokens'])
                        save_gzip(out / 'episodes' / (key + '.json.gz'), row)
                        audit = runner.audit_run(out, plan, job)
                        if not audit['all_pass']:
                            raise RuntimeError('Frozen independent replay audit failed')
                        checks.append({'policy': policy, 'hazard': hazard, 'malformed': name,
                            'all_pass': True, 'audit': str(out / 'audit.json'),
                            'clinical_success_required_for_transport_validation': False})
                        print(json.dumps({'synthetic_audits_passed': len(checks), 'total': 28}), flush=True)
        if source_hashes() != sources:
            raise RuntimeError('Repair source changed during validation')
        atomic_json(root / 'complete.json', {'all_pass': True, 'source_hashes': sources,
            'unit_tests': result.testsRun, 'old_crashes_reproduced': old_reproductions,
            'positive_path_equivalence': equivalence, 'checks': checks, 'time': time.time(),
            'held_out_tasks_used': False, 'model_inference': False, 'inference_api': False,
            'clinical_accuracy_not_a_gate': True})
    except BaseException:
        atomic_json(root / 'failure.json', {'error': traceback.format_exc(), 'time': time.time(),
            'completed_checks': checks, 'equivalence': equivalence})
        raise


if __name__ == '__main__':
    main()
