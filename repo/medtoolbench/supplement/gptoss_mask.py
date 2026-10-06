"""Narrow Torch 2.8 decode-mask compiler workaround; mask semantics unchanged.

The failing synthetic CUDA trace points to the compiled create_block_mask graph
for Q_LEN=1, inside sliding-window mask creation. Use PyTorch's same mask builder
without compilation for that shape only. Attention and MXFP4 kernels are unchanged.
Installed only inside the dedicated GPT-OSS process, never in other model runners.
"""
from functools import wraps


def decode_safe_builder(original):
    @wraps(original)
    def build(*args, **kwargs):
        q_length = kwargs.get('Q_LEN', args[3] if len(args) > 3 else None)
        if q_length == 1:
            if len(args) > 7:
                args = (*args[:7], False, *args[8:])
            else:
                kwargs['_compile'] = False
        return original(*args, **kwargs)
    build._medtoolbench_decode_mask_repair = True
    return build


def install_decode_mask_repair():
    import torch
    from transformers import masking_utils
    if torch.__version__.split('+')[0] != '2.8.0':
        raise RuntimeError('Decode mask repair is validated only for pinned Torch 2.8.0')
    if not getattr(masking_utils.create_block_mask, '_medtoolbench_decode_mask_repair', False):
        masking_utils.create_block_mask = decode_safe_builder(masking_utils.create_block_mask)
    return {'scope': 'Q_LEN=1 block-mask construction only', 'compile_decode_block_mask': False,
        'prefill_block_mask_and_attention_unchanged': True, 'torch': torch.__version__}
