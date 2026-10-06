"""Successful synthetic native read/write/readback, never a clinical test gate.

Requires actual successful server operations and an answer dependent on unseen
fixture values, not merely a tool name or an apparent terminal response.
"""
import argparse
import copy
import json
import os
import secrets
import time
import traceback
from collections import Counter
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import AllocatedGpuLease, validate_worker
from medtoolbench.research.fhir import HapiProcess, normalized_state
from medtoolbench.research.run import Episode
from medtoolbench.supplement.gptoss_native import advance, decode, VERSION, RECIPIENT
from medtoolbench.supplement.gptoss_native_model import NativeModel
from medtoolbench.supplement.gptoss_native_pipeline import source_hashes, ROOT
from medtoolbench.supplement.input_guard import install
from medtoolbench.supplement.runner import load_plan, public_messages
from medtoolbench.supplement.storage_recovery import atomic_json
from medtoolbench.supplement.transport import CallBudget, FactorClient


def native(action, recipient=RECIPIENT):
    return ' to=' + recipient + '<|channel|>commentary json<|message|>' + json.dumps(action) + '<|call|>'


def generation(raw):
    text, status, metadata = decode(raw)
    return {'raw_harmony': raw, 'text': text, 'harmony_status': status, 'native_transport': metadata}


def episode(store, policy, prompt):
    e = object.__new__(Episode)
    e.trace = []
    def call(method, path, body=None, actor='agent'):
        response = store.call(method, path, body)
        e.trace.append({'method': method, 'path': path, 'body': copy.deepcopy(body),
                        'actor': actor, 'response': copy.deepcopy(response)})
        return response
    e.budget = CallBudget(call, 48)
    e.client = FactorClient(e.budget, policy)
    e.messages = public_messages({'prompt': prompt}, policy, {'raw_call_budget': 48})
    e.generations, e.parsed_actions = [], []
    e.answer, e.done, e.stop = {}, False, 'turn_limit'
    return e


def positive(e, family, value, state):
    successful = [r for r in e.trace if 200 <= r['response']['status'] < 300]
    posts = [r for r in successful if r['method'] == 'POST' and r['path'] == '/Observation']
    if not e.done or len(posts) != 1:
        return False
    created = posts[0]['response']['body']
    identifier = created.get('id')
    reads = {r['path']: r for r in successful if r['method'] == 'GET'}
    expected = '/Observation/' + str(identifier)
    return (e.answer == {'family': family, 'value': value + 1, 'id': identifier} and
        {'/Patient/native-probe', '/Observation/native-source', expected} <= set(reads) and
        reads[expected]['response']['body'].get('valueQuantity', {}).get('value') == value + 1 and
        any(r.get('resourceType') == 'Observation' and r.get('id') == identifier and
            r.get('valueQuantity', {}).get('value') == value + 1 and
            r.get('subject', {}).get('reference') == 'Patient/native-probe' for r in state))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--worker', default='agb03-gpu0')
    a = p.parse_args(); a.out = a.out.resolve()
    if a.out.exists():
        raise RuntimeError('Never overwrite synthetic proof')
    a.out.mkdir(parents=True)
    log = (a.out / 'probe.log').open('x')
    os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
    sources = source_hashes()
    try:
        plan = load_plan(ROOT / 'data/supplement_a_v3/plan.json')
        worker = validate_worker(plan, a.worker)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(worker['gpu'])
        install()
        with AllocatedGpuLease(worker):
            atomic_json(a.out / 'status.json', {'state': 'loading', 'worker': worker, 'time': time.time()})
            model = NativeModel(ROOT / 'data/models/gpt-oss-20b')
            results = []
            with HapiProcess(plan['image'], a.out / 'hapi', heap='2g') as store:
                deterministic = []
                for verb, path, body in [
                    ('PUT', '/Patient/native-created', {'resourceType': 'Patient', 'id': 'native-created', 'active': True}),
                    ('GET', '/Patient/native-created', None),
                    ('POST', '/Patient', {'resourceType': 'Patient', 'active': False}),
                    ('DELETE', '/Patient/native-created', None)]:
                    e = episode(store, 'vanilla', 'Synthetic transport validation.')
                    action = {'method': verb, 'path': path}
                    if body is not None:
                        action['body'] = body
                    advance(e, generation(native(action)))
                    response = json.loads(e.messages[-1]['content'])
                    if not 200 <= response['status'] < 300 or e.parsed_actions != [action]:
                        raise RuntimeError('Deterministic live FHIR dispatch failed')
                    deterministic.append({'action': action, 'response': response, 'trace': e.trace})
                negatives = []
                for raw in [native({'method': 'POST', 'path': '/Observation', 'body': {'resourceType': 'Observation', 'meta': None}}),
                            native({'method': 'POST', 'path': '/Observation', 'body': 'invalid'}),
                            native({'path': '/Patient/native-created'}),
                            native({'method': 'DELETE', 'path': '/Patient/native-created'}, 'functions.unknown')]:
                    before = normalized_state(store.snapshot())
                    e = episode(store, 'vanilla', 'Synthetic negative transport validation.')
                    g = generation(raw); advance(e, g)
                    response = json.loads(e.messages[-1]['content'])
                    if response['status'] < 400 or before != normalized_state(store.snapshot()) or e.trace:
                        raise RuntimeError('Invalid arguments were repaired or executed')
                    negatives.append({'generation': g, 'messages': e.messages, 'state_unchanged': True})
                atomic_json(a.out / 'deterministic.json', {'positive': deterministic, 'negative': negatives})
                for policy in ('vanilla', 'medguard'):
                    family, value = 'Probe-' + secrets.token_hex(6), secrets.randbelow(25) + 5
                    fixture = [{'resourceType': 'Patient', 'id': 'native-probe', 'name': [{'family': family}]},
                        {'resourceType': 'Observation', 'id': 'native-source', 'status': 'final',
                         'code': {'text': 'synthetic source'}, 'subject': {'reference': 'Patient/native-probe'},
                         'valueQuantity': {'value': value}}]
                    store.reload(fixture)
                    prompt = ('Synthetic interface test, not a clinical decision: GET /Patient/native-probe for its name.family, '
                        'and GET /Observation/native-source for its valueQuantity.value. POST exactly one new Observation '
                        'to /Observation with status final, code {"text":"synthetic copy"}, '
                        'subject {"reference":"Patient/native-probe"}, and valueQuantity.value equal to the retrieved value plus 1. '
                        'Read the newly created Observation by its returned id to confirm it. Finish with '
                        '{"final":{"family":<retrieved family string>,"value":<confirmed numeric value>,"id":<created id string>}}. '
                        'Do not invent the hidden values or return the final answer before the successful write and readback.')
                    e = episode(store, policy, prompt)
                    for turn in range(12):
                        if e.done or e.budget.used >= 48:
                            break
                        atomic_json(a.out / 'status.json', {'state': 'read_write_readback', 'policy': policy,
                            'turn': turn, 'worker': worker, 'time': time.time()})
                        tokens = model.prompt(e.messages)
                        if len(tokens) + 2048 > 16384:
                            raise RuntimeError('Synthetic probe exceeded original context budget')
                        advance(e, model.generate_sample([tokens], 2048, 0)[0])
                        atomic_json(a.out / (policy + '_live.json'), {'messages': e.messages, 'generations': e.generations,
                            'answer': e.answer, 'done': e.done, 'trace': e.trace, 'raw_calls': e.budget.used})
                    state = store.snapshot()
                    row = {'policy': policy, 'fixture': fixture, 'state': state, 'messages': e.messages,
                        'generations': e.generations, 'trace': e.trace, 'answer': e.answer,
                        'done': e.done, 'raw_calls': e.budget.used, 'positive_path_pass': positive(e, family, value, state)}
                    results.append(row)
                    atomic_json(a.out / 'model_roundtrips.json', results)
                    if not row['positive_path_pass']:
                        raise RuntimeError('Actual successful read/write/readback required; error-only or guessed-final traces cannot pass')
                    cursor = 2
                    for g in e.generations:
                        tokens = model.prompt(e.messages[:cursor])
                        if digest(tokens) != g['native_prompt_hash'] or len(tokens) != g['input_tokens']:
                            raise RuntimeError('Native prompt audit failed')
                        cursor += 1 if g['native_transport']['kind'] == 'final' else 2
                if sources != source_hashes():
                    raise RuntimeError('Native source changed during synthetic proof')
                atomic_json(a.out / 'complete.json', {'all_pass': True, 'version': VERSION, 'source_hashes': sources,
                    'held_out_tasks_used': False, 'successful_read_write_feedback_conversations': len(results),
                    'negative_calls_preserve_payloads': True, 'model': model.metadata, 'worker': worker,
                    'time': time.time(), 'inference_api': False, 'clinical_accuracy_gate': False,
                    'statuses': dict(Counter(g['harmony_status'] for r in results for g in r['generations']))})
    except BaseException:
        atomic_json(a.out / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
