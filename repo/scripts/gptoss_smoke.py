"""Real GPU protocol and full-budget stress checks; no held-out tasks."""
import argparse
import json
import os
import time
import traceback
from pathlib import Path
from medtoolbench.research.cluster_execution import AllocatedGpuLease, validate_worker
from medtoolbench.supplement.runner import load_plan
from medtoolbench.supplement.gptoss_model import GptOssModel
from medtoolbench.supplement.storage_recovery import atomic_json
from medtoolbench.storage import digest

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--model', type=Path, default=Path('data/models/gpt-oss-20b'))
a = p.parse_args()
if a.out.exists():
    raise RuntimeError('Never overwrite preflight evidence')
a.out.mkdir(parents=True)
source = Path('repo/medtoolbench/supplement/gptoss_model.py').resolve()
adapter_hash = digest(source.read_text())
diagnostic_only = os.environ.get('CUDA_LAUNCH_BLOCKING', '0') != '0'
plan = load_plan(Path('data/supplement_a_v3/plan.json'))
lease = None
deadline = time.monotonic() + 7200
while lease is None:
    for worker in plan['workers']:
        validate_worker(plan, worker['name'])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(worker['gpu'])
        candidate = AllocatedGpuLease(worker)
        try:
            lease = candidate.__enter__()
            break
        except BlockingIOError:
            pass
    if lease is None:
        if time.monotonic() > deadline:
            raise RuntimeError('No free GPU in two-hour acquisition window')
        atomic_json(a.out / 'status.json', {'state': 'waiting_for_natural_gpu_release', 'time': time.time()})
        time.sleep(15)
try:
    atomic_json(a.out / 'status.json', {'state': 'loading', 'worker': worker, 'time': time.time()})
    model = GptOssModel(a.model)
    prompts = [model.prompt([{'role': 'user', 'content': 'Reply with exactly {"ready":true}, no other text.'}]) for _ in range(2)]
    rows = model.generate(prompts, 512)
    atomic_json(a.out / 'short.json', {'generations': rows, 'model': model.metadata})
    if not all(r['harmony_status'] == 'final' and json.loads(r['text']) == {'ready': True} for r in rows):
        raise RuntimeError('Basic Harmony/JSON smoke failed; inspect raw output')
    atomic_json(a.out / 'status.json', {'state': 'short_passed', 'worker': worker, 'time': time.time()})
    # Cover the unchanged maximum context and generation budgets, not new tasks.
    stress = []
    for output_budget in (2048, 4096):
        padding = model.tokenizer.encode(' clinical record', add_special_tokens=False)
        length = 16384 - output_budget
        prefix = prompts[0][:-2]
        tokens = prefix + (padding * (length // len(padding) + 1))[:length - len(prefix) - 2] + prompts[0][-2:]
        atomic_json(a.out / 'status.json', {'state': 'stress_running', 'worker': worker,
            'input_tokens_per_item': len(tokens), 'output_budget': output_budget, 'time': time.time()})
        model.torch.cuda.reset_peak_memory_stats()
        # min_new_tokens prevents an early EOS from making the memory test trivial.
        previous = model.model.generation_config.min_new_tokens
        model.model.generation_config.min_new_tokens = output_budget
        try:
            result = model.generate([tokens, tokens], output_budget)
        finally:
            model.model.generation_config.min_new_tokens = previous
        row = {'input_tokens_per_item': len(tokens), 'output_budget': output_budget,
            'output_tokens': [r['output_tokens'] for r in result],
            'seconds': sum(r['batch_seconds_per_item'] for r in result),
            'peak_allocated': model.torch.cuda.max_memory_allocated(),
            'peak_reserved': model.torch.cuda.max_memory_reserved()}
        stress.append(row)
        atomic_json(a.out / ('stress_' + str(output_budget) + '.json'), row)
        if any(r['output_tokens'] != output_budget for r in result):
            raise RuntimeError('Stress generation ended before its declared budget')
    if digest(source.read_text()) != adapter_hash:
        raise RuntimeError('Adapter changed during preflight; do not accept this proof')
    atomic_json(a.out / 'complete.json', {'all_pass': True, 'model': model.metadata, 'worker': worker,
        'adapter_hash': adapter_hash, 'stress': stress, 'time': time.time(),
        'counts_toward_results': False, 'held_out_tasks_used': False,
        'diagnostic_only': diagnostic_only})
    atomic_json(a.out / 'status.json', {'state': 'passed', 'worker': worker, 'time': time.time()})
    print(json.dumps({'all_pass': True, 'out': str(a.out), 'stress': stress}), flush=True)
except BaseException:
    atomic_json(a.out / 'failure.json', {'error': traceback.format_exc(), 'adapter_hash': adapter_hash,
        'counts_toward_results': False, 'time': time.time()})
    atomic_json(a.out / 'status.json', {'state': 'failed', 'worker': worker, 'time': time.time()})
    raise
finally:
    lease.__exit__()
