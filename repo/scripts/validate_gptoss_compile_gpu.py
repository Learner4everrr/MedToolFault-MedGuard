"""Synthetic GPU checks, independent of held-out clinical tasks and outcomes."""
import argparse
import importlib
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from medtoolbench.storage import digest
from medtoolbench.research.cluster_execution import AllocatedGpuLease, validate_worker
from medtoolbench.supplement.gptoss_compile import install, repaired_model_class
from medtoolbench.supplement.gptoss_mask import install_decode_mask_repair
from medtoolbench.supplement.runner import load_plan
from medtoolbench.supplement.storage_recovery import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args(); a.out = a.out.resolve()
    if a.out.exists():
        raise RuntimeError('Never overwrite validation evidence')
    a.out.mkdir(parents=True)
    with (a.out / 'probe.log').open('x') as log:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        try:
            run(a)
        except BaseException:
            atomic_json(a.out / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
            raise


def run(a):
    source = Path('repo/medtoolbench/supplement/gptoss_compile.py').resolve()
    sources = {str(f): digest(f.read_text()) for f in (source, Path(__file__).resolve())}
    plan = load_plan(Path('data/gptoss_native_replay_recovery_v1/campaign_a/plan.json'))
    worker = next(w for w in plan['workers'] if w['gpu'] == a.gpu)
    validate_worker(plan, worker['name'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    cache = Path('data/gptoss_cache').resolve()
    os.environ.update(TRITON_CACHE_DIR=str(cache / 'triton'), TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
                      TORCHINDUCTOR_COMPILE_THREADS='4')
    with AllocatedGpuLease(worker):
        import torch
        from torch._dynamo.utils import counters
        from torch.nn.attention.flex_attention import create_mask
        from transformers.masking_utils import flex_attention_mask, causal_mask_function, sliding_window_causal_mask_function
        from transformers.integrations.flex_attention import flex_attention_forward
        torch.set_num_threads(4); torch.manual_seed(20260925)
        repair = install(); install_decode_mask_repair()
        # If any attention call uses the old OOM-producing dense fallback, fail
        # before it can allocate. This guard is validation-only, not a new kernel.
        hop = importlib.import_module('torch._higher_order_ops.flex_attention')
        original_math = hop.math_attention
        def forbidden_dense(*args, **kwargs):
            raise RuntimeError('Unexpected dense Flex Attention fallback')
        hop.math_attention = forbidden_dense
        rows = []
        for qlen, klen, offset in [(1, 1, 0), (1, 129, 0), (1, 128, 14080),
                                  (127, 127, 0), (128, 128, 0), (129, 129, 0), (513, 513, 0)]:
            for batch in (1, 2):
                for window in (None, 128):
                    qoffset = offset + klen - qlen
                    padding = torch.ones((batch, offset+klen), dtype=torch.bool, device='cuda')
                    if batch == 2 and klen > 1:
                        padding[0, :offset+min(klen//3, 17)] = False
                    factory = causal_mask_function if window is None else sliding_window_causal_mask_function(window)
                    mask = flex_attention_mask(batch_size=batch, q_length=qlen, kv_length=klen,
                        q_offset=qoffset, kv_offset=offset, mask_function=factory, attention_mask=padding, device='cuda')
                    exact = create_mask(mask.mask_mod, batch, 1, qlen, klen, device='cuda')
                    qs = torch.arange(qlen, device='cuda')[:,None] + qoffset
                    ks = torch.arange(klen, device='cuda')[None,:] + offset
                    allowed = ks <= qs
                    if window is not None:
                        allowed &= ks > qs-window
                    expected = allowed[None,None,:,:] & padding[:,None,None,offset:offset+klen]
                    if not torch.equal(exact, expected):
                        raise RuntimeError('Token visibility mismatch')
                    query = torch.randn((batch, 4, qlen, 64), device='cuda', dtype=torch.bfloat16)
                    key = torch.randn((batch, 2, klen, 64), device='cuda', dtype=torch.bfloat16)
                    value = torch.randn_like(key); sinks = torch.randn(4, device='cuda', dtype=torch.bfloat16)
                    actual, _ = flex_attention_forward(SimpleNamespace(training=False), query, key, value,
                        mask, scaling=.125, s_aux=sinks)
                    logits = query.float() @ key.repeat_interleave(2, dim=1).float().transpose(-1,-2) * .125
                    logits = logits.masked_fill(~expected, float('-inf'))
                    logits = torch.cat([logits, sinks.float()[None,:,None,None].expand(batch,-1,qlen,1)], -1)
                    reference = (logits.softmax(-1)[...,:-1] @ value.repeat_interleave(2,dim=1).float()).transpose(1,2)
                    # Fully left-padded query rows are ignored by the model;
                    # compare real query positions, including all decode queries.
                    valid = padding[:,qoffset:qoffset+qlen]
                    torch.testing.assert_close(actual.float()[valid], reference[valid], atol=.03, rtol=.03)
                    rows.append({'qlen': qlen, 'klen': klen, 'offset': offset, 'batch': batch, 'window': window,
                        'mask_exact': True, 'max_abs_error': (actual.float()[valid]-reference[valid]).abs().max().item()})
                    atomic_json(a.out/'progress.json', {'phase': 'attention', 'checks': len(rows), 'time': time.time()})
                    print(rows[-1], flush=True)
        del query, key, value, mask, exact, actual, logits, reference
        torch.cuda.empty_cache()
        model = repaired_model_class()(Path('data/models/gpt-oss-20b'))
        generations = []
        # Explicit maximum input lengths for every formal generation budget.
        # Both ragged batches and full-budget generation (EOS disabled ONLY in
        # this synthetic stress probe) are covered without clinical test tasks.
        for length, limit, full_budget in [(512, 32, False), (768, 32, False), (1024, 32, False),
                (15360, 1024, False), (14336, 2048, False), (12288, 4096, True)]:
            base = model.prompt([{'role':'system','content':'Synthetic infrastructure test.'},
                {'role':'user','content':'Return a final JSON answer with the number 7.'}])
            token = model.tokenizer.encode(' test', add_special_tokens=False)[0]
            # Insert harmless synthetic fill before the final assistant header.
            tokens = base[:-3] + [token]*(length-len(base)) + base[-3:]
            prompts = [tokens, base]
            old_eos = model.model.generation_config.eos_token_id
            if full_budget:
                model.model.generation_config.eos_token_id = None
            torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
            output = model.generate_sample(prompts, limit, 0)
            model.model.generation_config.eos_token_id = old_eos
            if any(g['input_tokens'] != len(prompt) or g['native_prompt_hash'] != digest(prompt)
                   for g,prompt in zip(output,prompts)):
                raise RuntimeError('Input tokens changed')
            generations.append({'input_lengths': [len(x) for x in prompts], 'limit': limit,
                'forced_full_budget': full_budget, 'seconds': time.monotonic()-started,
                'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                'outputs': output})
            atomic_json(a.out/'model_stress.json', generations)
            atomic_json(a.out/'progress.json', {'phase':'model', 'checks':len(generations), 'time':time.time()})
            print({k:v for k,v in generations[-1].items() if k != 'outputs'}, flush=True)
        hop.math_attention = original_math
        if sources != {str(f):digest(f.read_text()) for f in (source,Path(__file__).resolve())}:
            raise RuntimeError('Validation sources changed')
        atomic_json(a.out/'complete.json', {'all_pass':True,'source_hashes':sources,'repair':repair,
            'attention_checks':rows,'model_stress_cases':len(generations),'dense_fallback_calls':0,
            'counters':{k:dict(v) for k,v in counters.items()},'model':model.metadata,
            'max_context':16384,'max_generation_budget':4096,'held_out_tasks_used':False,
            'clinical_accuracy_gate':False,'counts_toward_results':False,'time':time.time()})


if __name__ == '__main__':
    main()
