"""Development-only paired Qwen kernel probe; never dispatches formal tasks."""
import argparse
import functools
import importlib.metadata
import inspect
import json
import os
import time
import traceback
from pathlib import Path

from medtoolbench.storage import digest
from medtoolbench.research.build import load_gzip
from medtoolbench.research.cluster_execution import AllocatedGpuLease, validate_worker
from medtoolbench.research.models import LocalModel, input_ids
from . import b_qwen_priority as original
from .memory_execution import verify_completion
from .storage_recovery import atomic_json, read

ROOT = original.ROOT
CONTROL = ROOT / 'data/qwen_kernel_probe_v1'
FAST = ROOT / 'env_qwen_fast/bin/python'


def source_hashes():
    paths = [Path(__file__), Path(__file__).with_name('qwen_kernel_window.py'),
             ROOT / 'repo/tests/test_qwen_kernel_probe.py']
    return {str(p): digest(p.read_text()) for p in paths}


def prompts_from_row(row, tokenizer):
    if row['split'] not in ('qualification', 'development', 'dev'):
        raise RuntimeError('Held-out rows forbidden for kernel validation')
    result, index = [], 0
    for pos, message in enumerate(row['messages']):
        if message['role'] != 'assistant':
            continue
        generation = row['generations'][index]
        if message['content'] != generation['text']:
            raise RuntimeError('Development history/generation mismatch')
        ids = input_ids(tokenizer.apply_chat_template(row['messages'][:pos],
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False))
        if len(ids) != generation['input_tokens']:
            raise RuntimeError('Original development tokenization changed')
        result.append({'key': row['key'], 'turn': index,
            'condition': row['condition'], 'hazard': row['hazard'], 'ids': ids})
        index += 1
    if index != len(row['generations']):
        raise RuntimeError('Missing development generations')
    return result


def choose_cases(candidates):
    # All 18 cells: longest recorded development prompt, never success-based.
    cells = sorted({(x['condition'], x['hazard']) for x in candidates})
    if len(cells) != 18:
        raise RuntimeError('Expected three policies by six development conditions')
    picked = [max((x for x in candidates if (x['condition'], x['hazard']) == cell),
                  key=lambda x: (len(x['ids']), x['key'], x['turn'])) for cell in cells]
    ordered = sorted(candidates, key=lambda x: (len(x['ids']), x['key'], x['turn']))
    picked += [ordered[round(q * (len(ordered)-1))] for q in (0, .2, .4, .6, .8, 1)]
    return [{'name': 'development-batch-' + str(i//4), 'prompts': picked[i:i+4],
             'max_new_tokens': 2048} for i in range(0, len(picked), 4)]


def prepare(control):
    if (control / 'manifest.json').exists():
        raise RuntimeError('Never overwrite a kernel probe')
    base, plan = original.load(original.CONTROL / 'manifest.json')
    from transformers import AutoTokenizer
    system = next(x for x in plan['systems'] if x['name'] == original.MODEL)
    tokenizer = AutoTokenizer.from_pretrained(system['path'], local_files_only=True)
    candidates, evidence = [], []
    for job in original.selected(plan, 'qualification'):
        result_path = original.PLAN.parent / 'jobs' / job['id'] / 'result.json'
        result = read(result_path)
        if not result or result['state'] != 'complete':
            raise RuntimeError('Certified original development required')
        out = Path(result['run'])
        proof = verify_completion(out, plan, job)
        evidence.append({'result': str(result_path), 'result_hash': digest(result),
                         'audit_hash': digest(proof)})
        for file in sorted((out / 'episodes').glob('*.json.gz')):
            row = load_gzip(file)
            candidates += prompts_from_row(row, tokenizer)
    cases = choose_cases(candidates)
    atomic_json(control / 'cases.json', cases)
    manifest = {'created_at': time.time(), 'source_hashes': source_hashes(),
        'base_manifest_hash': base['manifest_hash'], 'plan_hash': plan['plan_hash'],
        'model': system, 'reference_python': plan['python'], 'fast_python': str(FAST),
        'cases_hash': digest(cases), 'development_evidence': evidence,
        'expected_versions': {'torch': '2.8.0+cu128', 'transformers': '5.17.0',
            'flash-linear-attention': '0.4.2', 'fla-core': '0.4.2', 'causal-conv1d': '1.6.0'},
        'held_out_tasks_used': False, 'counts_toward_benchmark': False,
        'formal_cutover_authorized': False,
        'selection': 'Longest prompt per development policy/environment cell plus six input-length quantiles; outcome-independent',
        'stress': {'batch_size': 4, 'prompt_tokens': 14336, 'new_tokens': 2048}}
    manifest['manifest_hash'] = digest(manifest)
    atomic_json(control / 'manifest.json', manifest)
    return manifest


def load(control):
    m = read(control / 'manifest.json')
    checksum = m.pop('manifest_hash')
    if digest(m) != checksum or m['source_hashes'] != source_hashes():
        raise RuntimeError('Kernel probe source or manifest changed')
    m['manifest_hash'] = checksum
    base, plan = original.load(original.CONTROL / 'manifest.json')
    if base['manifest_hash'] != m['base_manifest_hash'] or plan['plan_hash'] != m['plan_hash']:
        raise RuntimeError('Original scientific provenance changed')
    cases = read(control / 'cases.json')
    if digest(cases) != m['cases_hash'] or m['held_out_tasks_used'] or m['formal_cutover_authorized']:
        raise RuntimeError('Invalid development-only scope')
    for e in m['development_evidence']:
        if digest(read(e['result'])) != e['result_hash']:
            raise RuntimeError('Original development evidence changed')
    return m, plan, cases


def implementations(function):
    """Inspect HF wrappers without installing or replacing scientific code."""
    seen, stack, found = set(), [function], []
    while stack:
        f = stack.pop()
        if id(f) in seen or not inspect.isfunction(f):
            continue
        seen.add(id(f))
        values = inspect.getclosurevars(f).nonlocals
        if 'implementation' in values:
            implementation = values['implementation']
            found.append({'module': implementation.__module__, 'name': implementation.__name__,
                          'optimized': bool(values.get('is_new_implementation'))})
        if hasattr(f, '__wrapped__'):
            stack.append(f.__wrapped__)
        stack += [v for v in values.values() if inspect.isfunction(v)]
    return found


def compare(reference, fast):
    if reference['cases_hash'] != fast['cases_hash']:
        raise RuntimeError('Cannot compare different development inputs')
    if [r['name'] for r in reference['cases']] != [r['name'] for r in fast['cases']]:
        raise RuntimeError('Incomplete paired comparison')
    differences, total = [], 0
    for a, b in zip(reference['cases'], fast['cases']):
        if len(a['outputs']) != len(b['outputs']):
            raise RuntimeError('Paired batch coverage changed')
        for index, (x, y) in enumerate(zip(a['outputs'], b['outputs'])):
            total += 1
            fields = ('text', 'input_tokens', 'output_tokens', 'generation_limit')
            if any(x[k] != y[k] for k in fields):
                differences.append({'batch': a['name'], 'index': index,
                    'reference_output_tokens': x['output_tokens'], 'fast_output_tokens': y['output_tokens'],
                    'reference_text_hash': digest(x['text']), 'fast_text_hash': digest(y['text'])})
    old_seconds = sum(r['seconds'] for r in reference['cases'])
    new_seconds = sum(r['seconds'] for r in fast['cases'])
    old_tokens = sum(o['output_tokens'] for r in reference['cases'] for o in r['outputs'])
    new_tokens = sum(o['output_tokens'] for r in fast['cases'] for o in r['outputs'])
    return {'paired_prompts': total, 'exact_match': total-len(differences), 'differences': differences,
        'all_outputs_identical': not differences, 'reference_seconds': old_seconds, 'fast_seconds': new_seconds,
        'wall_time_ratio_not_equal_work_if_outputs_differ': old_seconds/new_seconds,
        'reference_output_tokens': old_tokens, 'fast_output_tokens': new_tokens,
        'reference_useful_tokens_per_second': old_tokens/old_seconds,
        'fast_useful_tokens_per_second': new_tokens/new_seconds,
        'formal_cutover_authorized': False,
        'next_step': 'Full paired development validation required' if not differences else 'Review backend numerical/trajectory differences before any formal migration'}


def run_model(control, backend, worker_name):
    m, plan, cases = load(control)
    out = control / backend
    if out.exists():
        raise RuntimeError('Never overwrite probe evidence')
    out.mkdir(parents=True)
    worker = validate_worker(plan, worker_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(worker['gpu'])
    try:
        with AllocatedGpuLease(worker):
            import torch
            import transformers
            if torch.__version__ != m['expected_versions']['torch'] or transformers.__version__ != m['expected_versions']['transformers']:
                raise RuntimeError('Core numerical runtime changed')
            if backend == 'fast':
                import fla
                import causal_conv1d
                for name in ('flash-linear-attention', 'fla-core', 'causal-conv1d'):
                    if importlib.metadata.version(name) != m['expected_versions'][name]:
                        raise RuntimeError('Unpinned kernel dependency: ' + name)
            from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
            bindings = {n: implementations(getattr(qwen, n)) for n in (
                'causal_conv1d_fn', 'causal_conv1d_update',
                'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule')}
            if any(len(v) != 1 or v[0]['optimized'] != (backend == 'fast') for v in bindings.values()):
                raise RuntimeError('Kernel dispatch is not the declared backend: ' + str(bindings))
            atomic_json(out / 'bindings.json', bindings)
            model = LocalModel(Path(m['model']['path']), thinking=False, seed=20260917)
            atomic_json(out / 'model.json', model.metadata)
            model.generate([cases[0]['prompts'][0]['ids'][:256]], 32)
            results = []
            for case in cases:
                torch.cuda.reset_peak_memory_stats()
                outputs = model.generate([p['ids'] for p in case['prompts']], case['max_new_tokens'])
                record = {'name': case['name'], 'outputs': outputs,
                    'seconds': sum(r['batch_seconds_per_item'] for r in outputs),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                results.append(record)
                atomic_json(out / (case['name'] + '.json'), record)
                atomic_json(out / 'progress.json', {'completed': len(results), 'total': len(cases), 'time': time.time()})
            report = {'backend': backend, 'cases_hash': m['cases_hash'], 'cases': results,
                'model': model.metadata, 'kernel_bindings': bindings, 'time': time.time(),
                'held_out_tasks_used': False, 'counts_toward_benchmark': False}
            atomic_json(out / 'comparison_inputs.json', report)
            if backend == 'fast':
                # Capacity stress only, never a clinical result or equality claim.
                src = cases[-1]['prompts'][-1]['ids']
                length = m['stress']['prompt_tokens']
                prompt = (src * ((length + len(src)-1)//len(src)))[:length]
                encoded = model.tokenizer.pad({'input_ids': [prompt]*4}, return_tensors='pt',
                    padding=True, return_attention_mask=True).to('cuda:0')
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                with torch.inference_mode():
                    generated = model.model.generate(**encoded, max_new_tokens=2048,
                        min_new_tokens=2048, do_sample=False, eos_token_id=None,
                        pad_token_id=model.tokenizer.pad_token_id, use_cache=True)
                torch.cuda.synchronize()
                if tuple(generated.shape) != (4, 16384):
                    raise RuntimeError('Incomplete declared long-context stress')
                atomic_json(out / 'stress.json', {'all_pass': True, 'shape': list(generated.shape),
                    'seconds': time.monotonic()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'counts_toward_benchmark': False})
            atomic_json(out / 'complete.json', {'all_pass': True, 'cases': len(results), 'time': time.time()})
    except BaseException:
        atomic_json(out / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('prepare', 'run'))
    p.add_argument('--control', type=Path, default=CONTROL)
    p.add_argument('--backend', choices=('reference', 'fast'))
    p.add_argument('--worker', default='agb03-gpu5')
    a = p.parse_args()
    if a.command == 'prepare':
        prepare(a.control.resolve())
    else:
        run_model(a.control.resolve(), a.backend, a.worker)
