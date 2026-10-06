"""Synthetic GPU mask/attention equivalence, including ragged batch decoding."""
import argparse
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from medtoolbench.supplement.storage_recovery import atomic_json
from medtoolbench.research.cluster_execution import AllocatedGpuLease, validate_worker
from medtoolbench.supplement.runner import load_plan
from medtoolbench.supplement.gptoss_mask import install_decode_mask_repair
from medtoolbench.storage import digest

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--gpu', type=int, default=6)
a = p.parse_args()
if a.out.exists():
    raise RuntimeError('Never overwrite a validation receipt')
a.out.mkdir(parents=True)
worker = next(w for w in load_plan(Path('data/supplement_a_v3/plan.json'))['workers'] if w['gpu'] == a.gpu)
validate_worker({'workers': [worker]}, worker['name'])
os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
cache = Path('data/gptoss_cache').resolve()
os.environ.update(TRITON_CACHE_DIR=str(cache / 'triton'), TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
                  TORCHINDUCTOR_COMPILE_THREADS='4')
with AllocatedGpuLease(worker):
    try:
        import torch
        from torch.nn.attention.flex_attention import create_mask
        from transformers.masking_utils import flex_attention_mask, causal_mask_function, sliding_window_causal_mask_function
        from transformers.integrations.flex_attention import flex_attention_forward
        torch.set_num_threads(4)
        torch.manual_seed(20260922)
        repair = install_decode_mask_repair()
        rows = []
        # Actual full/sliding decode, block boundaries, and offsets from the failed run.
        for length, offset, window in [(1, 0, None), (127, 0, None), (128, 0, None),
                (129, 0, None), (12289, 0, None), (16384, 0, None),
                (128, 12161, 128), (128, 14209, 128), (128, 16256, 128)]:
            for ragged in (False, True):
                q_offset = offset + length - 1
                padding = torch.ones((2, q_offset+1), dtype=torch.bool, device='cuda')
                if ragged:
                    padding[0, :offset + max(0, length//2)] = False
                factory = causal_mask_function if window is None else sliding_window_causal_mask_function(window)
                mask = flex_attention_mask(batch_size=2, q_length=1, kv_length=length,
                    q_offset=q_offset, kv_offset=offset, mask_function=factory,
                    attention_mask=padding, device='cuda')
                exact = create_mask(mask.mask_mod, 2, 1, 1, length, device='cuda')
                positions = torch.arange(length, device='cuda') + offset
                allowed = (positions <= q_offset)
                if window is not None:
                    allowed &= positions > q_offset-window
                expected = allowed[None,None,None,:] & padding[:,None,None,offset:offset+length]
                if not torch.equal(exact, expected):
                    raise RuntimeError('Token-level attention visibility changed')
                query = torch.randn((2, 4, 1, 64), device='cuda', dtype=torch.bfloat16)
                key = torch.randn((2, 2, length, 64), device='cuda', dtype=torch.bfloat16)
                value = torch.randn_like(key)
                sinks = torch.randn(4, device='cuda', dtype=torch.bfloat16)
                actual, _ = flex_attention_forward(SimpleNamespace(training=False), query, key, value,
                    mask, scaling=0.125, s_aux=sinks)
                expanded_key = key.repeat_interleave(2, dim=1)
                expanded_value = value.repeat_interleave(2, dim=1)
                logits = (query.float() @ expanded_key.float().transpose(-1,-2)) * 0.125
                logits = logits.masked_fill(~expected, float('-inf'))
                logits = torch.cat((logits, sinks.float()[None,:,None,None].expand(2,-1,1,1)), dim=-1)
                reference = (logits.softmax(dim=-1)[...,:-1] @ expanded_value.float()).transpose(1,2)
                error = (actual.float()-reference).abs().max().item()
                torch.testing.assert_close(actual.float(), reference, atol=0.03, rtol=0.03)
                rows.append({'kv_length': length, 'kv_offset': offset, 'window': window,
                    'ragged': ragged, 'mask_exact': True, 'max_abs_attention_error': error})
                atomic_json(a.out / 'progress.json', {'checks': rows, 'time': time.time()})
                print(rows[-1], flush=True)
        atomic_json(a.out / 'complete.json', {'all_pass': True, 'repair': repair, 'checks': rows,
            'repair_hash': digest(Path('repo/medtoolbench/supplement/gptoss_mask.py').read_text()),
            'held_out_tasks_used': False, 'counts_toward_results': False, 'time': time.time()})
    except BaseException:
        atomic_json(a.out / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        raise
