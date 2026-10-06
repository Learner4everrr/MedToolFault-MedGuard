"""Isolated native GPT-OSS A/C/D runner, preserving existing scientific code."""
import argparse
import importlib
import sys
from collections import Counter
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.build import load_gzip
from .gptoss_native import advance, decode, render, VERSION
from .gptoss_native_model import NativeModel
from .gptoss_model import FROZEN_DATE
from .input_guard import install
from .runner import load_plan
from .storage_recovery import atomic_json


def audit_native(out, plan):
    import json
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(plan['systems'][0]['path'], local_files_only=True)
    template = tok.chat_template.replace('strftime_now("%Y-%m-%d")', json.dumps(FROZEN_DATE))
    counts, generations, rows = Counter(), 0, 0
    for file in sorted((out / 'episodes').glob('*.json.gz')):
        row = load_gzip(file)
        rows += 1
        cursor = 2
        for g in row['generations']:
            if decode(g['raw_harmony']) != (g['text'], g['harmony_status'], g['native_transport']):
                raise RuntimeError('Native raw-stream replay mismatch: ' + str(file))
            expected = render(tok, template, row['messages'][:cursor])
            if len(expected) != g['input_tokens'] or digest(expected) != g['native_prompt_hash']:
                raise RuntimeError('Native input/history replay mismatch: ' + str(file))
            message = row['messages'][cursor]
            if message != {'role': 'assistant', 'content': g['text'], 'native_transport': g['native_transport']}:
                raise RuntimeError('Native history annotation mismatch')
            cursor += 1
            if cursor < len(row['messages']) and row['messages'][cursor]['role'] == 'user':
                cursor += 1
            counts[g['harmony_status']] += 1
            generations += 1
        if cursor != len(row['messages']):
            raise RuntimeError('Unexpected conversation tail')
    if not rows:
        raise RuntimeError('No episode evidence')
    proof = {'all_pass': True, 'version': VERSION, 'plan_hash': plan['plan_hash'],
        'generations': generations, 'episodes': rows, 'statuses': dict(counts), 'prompt_replay_all_generations': True,
        'clinical_accuracy_gate': False, 'model_format_errors_preserved': True}
    atomic_json(out / 'native_audit.json', proof)
    return proof


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--native-manifest', type=Path, required=True)
    args, rest = p.parse_known_args()
    from .gptoss_native_pipeline import load, development_passed
    manifest = load(args.native_manifest)
    cp = Path(rest[rest.index('--plan') + 1]).resolve()
    if str(cp) not in manifest['campaigns']:
        raise RuntimeError('Plan outside GPT-OSS A/C/D scope')
    plan = load_plan(cp)
    if plan['plan_hash'] != manifest['plan_hashes'][str(cp)]:
        raise RuntimeError('Native plan identity changed')
    job = next(j for j in plan['jobs'] if j['id'] == rest[rest.index('--job') + 1])
    if job['model'] != 'gptoss20b' or {s['name'] for s in plan['systems']} != {'gptoss20b'}:
        raise RuntimeError('Native runner permits only GPT-OSS-20B')
    if job['stage'] == 'test' and not development_passed(manifest):
        raise RuntimeError('All 260 fresh development episodes and native audits must precede formal work')
    from . import runner, corrected_runner
    from .concurrent_repair import ConcurrentRejectionProxy
    install()
    parent = runner.SupplementEpisode

    class NativeEpisode(parent):
        def advance(self, generation):
            advance(self, generation)

    class RecordedModel(NativeModel):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.metadata['native_manifest_hash'] = manifest['manifest_hash']

    original_write = runner.write_json
    def recorded_write(path, value):
        if Path(path).name == 'config.json':
            value = {**value, 'native_amendment': plan['native_amendment'],
                     'native_manifest_hash': manifest['manifest_hash']}
        return original_write(path, value)
    runner.SupplementEpisode, runner.SampleModel, runner.write_json = NativeEpisode, RecordedModel, recorded_write
    if job['settings'].get('stress', {}).get('concurrent'):
        if not plan['runner_module'].endswith('corrected_runner'):
            raise RuntimeError('Unexpected concurrent runner')
        corrected_runner.InterfaceConsistentProxy = ConcurrentRejectionProxy
    module = importlib.import_module(plan['runner_module'])
    sys.argv = [module.__file__, *rest]
    module.main()
    audit_native(Path(rest[rest.index('--out') + 1]), plan)


if __name__ == '__main__':
    main()
