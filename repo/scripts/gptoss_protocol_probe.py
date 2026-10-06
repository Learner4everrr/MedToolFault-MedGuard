"""Real local model + HAPI + restricted-tool roundtrips on synthetic fixtures.

No benchmark task, reference answer, test split or clinical accuracy gate.
The frozen clinical prompts/advance methods are exercised without modification.
"""
import argparse
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import AllocatedGpuLease, validate_worker
from medtoolbench.research.fhir import HapiProcess
from medtoolbench.research.protocol import SYSTEM, parse_action
from medtoolbench.research.run import Episode as FhirEpisode
from medtoolbench.supplement.external_runner import Episode as ExternalEpisode, parse
from medtoolbench.supplement.external_tools import ExternalTools, Retrieval
from medtoolbench.supplement.gptoss_protocol import decode
from medtoolbench.supplement.gptoss_protocol_model import GptOssProtocolModel
from medtoolbench.supplement.input_guard import install
from medtoolbench.supplement.runner import load_plan
from medtoolbench.supplement.storage_recovery import atomic_json


def hashes(root):
    names = ['repo/medtoolbench/supplement/' + name + '.py' for name in
        ('gptoss_model', 'gptoss_mask', 'gptoss_protocol', 'gptoss_protocol_model', 'input_guard')]
    names.append('repo/scripts/gptoss_protocol_probe.py')
    return {str(root / name): digest((root / name).read_text()) for name in names}


def native(name, arguments):
    return '<|channel|>commentary to=' + name + '<|constrain|>json<|message|>' + json.dumps(arguments) + '<|call|>'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--worker', default='agb03-gpu0')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    if a.out.exists():
        raise RuntimeError('Never overwrite protocol validation')
    a.out.mkdir(parents=True)
    log = (a.out / 'probe.log').open('x')
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sources = hashes(root)
    plan = load_plan(root / 'data/supplement_a_v3/plan.json')
    worker = validate_worker(plan, a.worker)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(worker['gpu'])
    install()
    try:
        with AllocatedGpuLease(worker):
            atomic_json(a.out / 'status.json', {'state': 'loading', 'worker': worker, 'time': time.time()})
            model = GptOssProtocolModel(root / 'data/models/gpt-oss-20b')
            with HapiProcess(plan['image'], a.out / 'hapi', heap='2g') as store:
                fixture = [
                    {'resourceType': 'Patient', 'id': 'protocol-probe', 'name': [{'family': 'Roundtrip'}]},
                    {'resourceType': 'Observation', 'id': 'probe-1', 'status': 'final',
                     'code': {'text': 'synthetic'}, 'subject': {'reference': 'Patient/protocol-probe'},
                     'valueQuantity': {'value': 7}},
                ]
                returned = store.reload(fixture)
                resources = {kind: [r for r in returned if r['resourceType'] == kind]
                             for kind in ('Patient', 'Encounter', 'Observation')}
                # Deterministic transport integration: all FHIR verbs travel
                # through the original parser and live, owned HAPI boundary.
                deterministic = []
                for name, action in [
                    ('repo_browser.get', {'method': 'GET', 'path': '/Patient/protocol-probe'}),
                    ('repo_browser.post', {'method': 'POST', 'path': '/Patient',
                                          'body': {'resourceType': 'Patient', 'active': True}}),
                    ('repo_browser.put', {'method': 'PUT', 'path': '/Patient/probe-created',
                                         'body': {'resourceType': 'Patient', 'id': 'probe-created', 'active': False}}),
                    ('repo_browser.delete', {'method': 'DELETE', 'path': '/Patient/probe-created'}),
                ]:
                    raw = native(name, action)
                    text, status, meta = decode(raw, 'fhir')
                    parsed = parse_action(text)
                    response = store.call(parsed['method'], parsed['path'], parsed.get('body'))
                    deterministic.append({'raw': raw, 'action': parsed, 'response': response, 'bridge': meta})
                    if status != 'native_fhir' or response['status'] >= 400:
                        raise RuntimeError('Live FHIR envelope roundtrip failed')
                store.reload(fixture)
                tools = ExternalTools(Retrieval(resources, 'protocol-probe', 'clean'), 'vanilla')
                for name, arguments in [('metadata', {}), ('get_resources', {
                    'resource_type': 'Observation', 'patient_fhir_id': 'protocol-probe'}),
                    ('execute_python_code', {'code': 'result = len(retrieved_resources["Observation"])'})]:
                    raw = native(name, arguments)
                    text, status, meta = decode(raw, 'external')
                    action = parse(text)
                    response = tools.call(action)
                    deterministic.append({'raw': raw, 'action': action, 'response': response, 'bridge': meta})
                    if status != 'native_external' or response['status'] != 200:
                        raise RuntimeError('Live restricted external-tool roundtrip failed')
                atomic_json(a.out / 'deterministic_roundtrips.json', deterministic)

                results = []
                for mode in ('fhir', 'external'):
                    model.protocol_mode = mode
                    episodes, traces = [], []
                    for _ in range(2):
                        trace = []
                        traces.append(trace)
                        if mode == 'fhir':
                            episode = object.__new__(FhirEpisode)
                            episode.messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content':
                                'Transport check on synthetic data: retrieve /Patient/protocol-probe, then return '
                                '{"final":{"family": "the family name from the actual response"}}. Do not guess without the read.'}]
                            class Client:
                                def __init__(self, trace):
                                    self.trace = trace
                                def call(self, method, path, body=None):
                                    response = store.call(method, path, body)
                                    self.trace.append({'method': method, 'path': path, 'body': body, 'response': response})
                                    return response
                            episode.client = Client(trace)
                            episode.generations, episode.parsed_actions = [], []
                            episode.answer, episode.done, episode.stop = {}, False, 'turn_limit'
                        else:
                            q = {'question': 'Retrieve all Observation resources for this synthetic patient with get_resources, '
                                 'then use execute_python_code to count them. Return the count as rows.',
                                 'assumption': 'This is an integration test, not a clinical question.',
                                 'patient_fhir_id': 'protocol-probe'}
                            episode = ExternalEpisode(q, {'hazard': 'clean', 'condition': 'vanilla'}, resources, {}, 'synthetic live HAPI')
                        episodes.append(episode)
                    for turn in range(8):
                        active = [e for e in episodes if not e.done]
                        if not active:
                            break
                        atomic_json(a.out / 'status.json', {'state': 'roundtrip_running', 'mode': mode, 'turn': turn,
                            'worker': worker, 'time': time.time()})
                        prompts = [model.prompt(e.messages) for e in active]
                        if any(len(p) + 2048 > 16384 for p in prompts):
                            raise RuntimeError('Synthetic probe exceeded original context budget')
                        for episode, generation in zip(active, model.generate_sample(prompts, 2048, 0)):
                            episode.advance(generation)
                        atomic_json(a.out / (mode + '_live.json'), [{'messages': e.messages, 'generations': e.generations,
                            'answer': e.answer, 'done': e.done} for e in episodes])
                    for index, episode in enumerate(episodes):
                        trace = traces[index] if mode == 'fhir' else episode.tools.events
                        row = {'mode': mode, 'messages': episode.messages, 'generations': episode.generations,
                               'answer': episode.answer, 'done': episode.done, 'trace': trace}
                        results.append(row)
                        if not trace or not episode.done:
                            raise RuntimeError('Real model did not complete a tool-feedback roundtrip')
                        # A malformed model emission is not an adapter failure.
                        # Replay must match exactly, including safe rejection and
                        # the original runner's 400 feedback. Do not gate on a
                        # zero model-format-error rate or "repair" its answer.
                        for g in episode.generations:
                            if decode(g['raw_harmony'], mode) != (g['text'], g['harmony_status'], g['protocol_bridge']):
                                raise RuntimeError('Raw native stream replay mismatch')
                        if not any(g['harmony_status'] == 'native_' + mode for g in episode.generations):
                            raise RuntimeError('A real native envelope must reach each dispatcher')
                        if mode == 'external' and not {'get_resources', 'execute_python_code'} <= {r['tool'] for r in trace}:
                            raise RuntimeError('Both retrieval and restricted code feedback must be exercised')
                atomic_json(a.out / 'model_roundtrips.json', results)
                if hashes(root) != sources:
                    raise RuntimeError('Protocol source changed during probe')
                atomic_json(a.out / 'complete.json', {'all_pass': True, 'source_hashes': sources,
                    'model': model.metadata, 'worker': worker, 'time': time.time(),
                    'synthetic_model_roundtrips': len(results), 'deterministic_tool_roundtrips': len(deterministic),
                    'model_generation_statuses': dict(Counter(g['harmony_status'] for r in results for g in r['generations'])),
                    'model_tool_response_statuses': dict(Counter(str(t['response']['status']) for r in results for t in r['trace'])),
                    'model_format_errors_preserved': True,
                    'counts_toward_results': False, 'held_out_tasks_used': False, 'inference_api': False,
                    'clinical_accuracy_gate': False, 'diagnostic_only': False})
                atomic_json(a.out / 'status.json', {'state': 'passed', 'time': time.time()})
    except BaseException:
        atomic_json(a.out / 'failure.json', {'error': traceback.format_exc(), 'source_hashes': sources,
                                           'time': time.time(), 'counts_toward_results': False})
        raise


if __name__ == '__main__':
    main()
