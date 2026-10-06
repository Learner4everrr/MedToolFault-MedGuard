"""Lossless post-save native audit. Inference and scientific code are unchanged.

The gzip writer sorts dictionary keys. Recreate transient transport metadata
from the preserved raw string, never from the sorted argument dictionary.
Check semantic equality before reconstruction, then exact input-token hashes.
"""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.build import load_gzip
from .gptoss_model import FROZEN_DATE
from .gptoss_native import VERSION, decode, render
from .storage_recovery import atomic_json, read

REVISION = 'raw-order-replay-audit-v1'


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evidence_hashes(out):
    files = sorted({*out.glob('*.json'), *out.glob('episodes/*.json.gz'), *out.glob('states/*.json.gz')})
    # A freshly created correction certificate cannot fingerprint itself.
    return {str(p.relative_to(out)): checksum(p) for p in files if p.name != 'native_replay_audit.json'}


def tokenizer_for(plan):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(plan['systems'][0]['path'], local_files_only=True)
    template = tok.chat_template.replace('strftime_now("%Y-%m-%d")', json.dumps(FROZEN_DATE))
    return tok, template


def audit_row(row, tokenizer, template):
    messages = copy.deepcopy(row['messages'])
    if len(messages) < 2:
        raise RuntimeError('Missing public messages')
    cursor, counts = 2, Counter()
    for generation in row['generations']:
        text, status, metadata = decode(generation['raw_harmony'])
        if ((text, status) != (generation['text'], generation['harmony_status']) or
                digest(metadata) != digest(generation['native_transport'])):
            raise RuntimeError('Raw native stream or semantic metadata mismatch')
        expected = render(tokenizer, template, messages[:cursor])
        if len(expected) != generation['input_tokens'] or digest(expected) != generation['native_prompt_hash']:
            raise RuntimeError('Exact native input replay mismatch after raw-order reconstruction')
        if cursor >= len(messages) or digest(messages[cursor]) != digest(
                {'role': 'assistant', 'content': text, 'native_transport': metadata}):
            raise RuntimeError('Native assistant history mismatch')
        # Pure in-memory reconstruction. No saved row, request or answer changes.
        messages[cursor]['native_transport'] = metadata
        cursor += 1
        if cursor < len(messages) and messages[cursor]['role'] == 'user':
            cursor += 1
        counts[status] += 1
    if cursor != len(messages):
        raise RuntimeError('Unexpected conversation tail')
    return counts


def audit_native(out, plan, destination=None, tokenizer=None):
    out = Path(out)
    destination = Path(destination) if destination else out / 'native_replay_audit.json'
    if destination.exists():
        raise RuntimeError('Never overwrite an audit certificate')
    config = read(out / 'config.json')
    if not config or config['plan_hash'] != plan['plan_hash']:
        raise RuntimeError('Audit plan/config identity mismatch')
    job = next(j for j in plan['jobs'] if j['id'] == config['job']['id'])
    if job != config['job']:
        raise RuntimeError('Audit job assignment changed')
    wanted = {e['key'] for e in job['episodes']}
    before = evidence_hashes(out)
    tok, template = tokenizer if tokenizer else tokenizer_for(plan)
    seen, counts = set(), Counter()
    for file in sorted((out / 'episodes').glob('*.json.gz')):
        row = load_gzip(file)
        if row['key'] in seen:
            raise RuntimeError('Duplicate audit episode')
        seen.add(row['key'])
        counts.update(audit_row(row, tok, template))
    if seen != wanted or not seen:
        raise RuntimeError('Incomplete native audit coverage')
    if evidence_hashes(out) != before:
        raise RuntimeError('Evidence changed during native audit')
    proof = {'all_pass': True, 'revision': REVISION, 'version': VERSION, 'plan_hash': plan['plan_hash'],
        'job': job['id'], 'run': str(out.resolve()), 'episodes': len(seen),
        'generations': sum(counts.values()), 'statuses': dict(counts), 'evidence_sha256': before,
        'prompt_replay_all_generations': True, 'exact_token_hashes_required': True,
        'semantic_metadata_required': True, 'clinical_accuracy_gate': False,
        'inference_changed': False, 'raw_evidence_modified': False, 'model_format_errors_preserved': True}
    atomic_json(destination, proof)
    return proof
