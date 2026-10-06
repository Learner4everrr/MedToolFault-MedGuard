"""Synthetic-only CUDA reproduction; never a qualification or formal result."""
import argparse
import os
import time
import traceback
from pathlib import Path

from medtoolbench.research.cluster_execution import AllocatedGpuLease, validate_worker
from medtoolbench.supplement.runner import load_plan
from medtoolbench.supplement.storage_recovery import atomic_json
from medtoolbench.supplement.gptoss_model import GptOssModel

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--lengths', type=int, nargs='+', default=[79, 14336, 12288])
p.add_argument('--outputs', type=int, nargs='+', default=[32])
p.add_argument('--sync-modules', action='store_true')
a = p.parse_args()
if a.out.exists():
    raise RuntimeError('Never overwrite diagnostic evidence')
a.out.mkdir(parents=True)
plan = load_plan(Path('data/supplement_a_v3/plan.json'))
worker = next(w for w in plan['workers'] if w['gpu'] == a.gpu)
validate_worker(plan, worker['name'])
os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
with AllocatedGpuLease(worker):
    try:
        atomic_json(a.out / 'status.json', {'state': 'loading', 'time': time.time()})
        model = GptOssModel(Path('data/models/gpt-oss-20b'))
        step = {'calls': 0}
        def progress(module, args, kwargs):
            step['calls'] += 1
            if step['calls'] == 1 or step['calls'] % 128 == 0:
                print(f'FORWARD calls={step["calls"]}', flush=True)
                atomic_json(a.out / 'decode_progress.json', {**step, 'time': time.time()})
        model.model.register_forward_pre_hook(progress, with_kwargs=True)
        if a.sync_modules:
            def sync_hook(name):
                def done(module, args, output):
                    try:
                        model.torch.cuda.synchronize()
                    except BaseException:
                        atomic_json(a.out / 'failing_module.json', {'module': name,
                            'calls': step['calls'], 'time': time.time()})
                        raise
                return done
            for name, module in model.model.named_modules():
                if type(module).__name__ in {'GptOssAttention', 'Mxfp4GptOssExperts'}:
                    module.register_forward_hook(sync_hook(name))
        prompt = model.prompt([{'role': 'user', 'content': 'Reply with exactly {"ready":true}, no other text.'}])
        padding = model.tokenizer.encode(' clinical record', add_special_tokens=False)
        prefix = prompt[:-2]
        rows = []
        for n, length in enumerate(a.lengths):
            output = a.outputs[min(n, len(a.outputs)-1)]
            tokens = prefix + (padding * (length // len(padding) + 1))[:length-len(prefix)-2] + prompt[-2:]
            atomic_json(a.out / 'status.json', {'state': 'generating', 'length': length,
                'output': output, 'time': time.time(), 'diagnostic_only': True})
            print(f'GENERATE input={len(tokens)} output={output} batch=2', flush=True)
            model.model.generation_config.min_new_tokens = output
            result = model.generate([tokens, tokens], output)
            row = {'input': length, 'output': output, 'rows': result, 'time': time.time()}
            rows.append(row)
            atomic_json(a.out / f'case_{n}.json', row)
            print(f'PASS input={length} output={output}', flush=True)
        atomic_json(a.out / 'complete.json', {'all_pass': True, 'cases': len(rows), 'diagnostic_only': True})
    except BaseException:
        atomic_json(a.out / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        raise
