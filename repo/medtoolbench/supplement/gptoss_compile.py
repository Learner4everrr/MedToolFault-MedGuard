"""Versioned local compiler-capacity repair; no model/protocol/budget changes.

Torch 2.8's eight-variant Dynamo cache silently falls back to dense attention
after the heterogeneous native masks exhaust it. Preserve the same Flex
Attention implementation, give its finite shape/mask variants room, and fail
closed instead of silently allocating a quadratic dense attention tensor.
"""
REVISION = 'flex-compiler-capacity-v1'
RECOMPILE_LIMIT = 256
ACCUMULATED_LIMIT = 4096


def configure(config):
    config.suppress_errors = False
    config.recompile_limit = RECOMPILE_LIMIT
    config.accumulated_recompile_limit = ACCUMULATED_LIMIT
    config.fail_on_recompile_limit_hit = True
    return {'revision': REVISION, 'recompile_limit': config.recompile_limit,
        'accumulated_recompile_limit': config.accumulated_recompile_limit,
        'fail_on_recompile_limit_hit': config.fail_on_recompile_limit_hit,
        'suppress_errors': config.suppress_errors,
        'attention_algorithm': 'unchanged Transformers Flex Attention',
        'weights_precision_batch_budgets_prompts_unchanged': True}


def install():
    import torch
    from torch._dynamo import config
    if torch.__version__.split('+')[0] != '2.8.0':
        raise RuntimeError('Compiler capacity repair requires pinned Torch 2.8.0')
    return configure(config)


def repaired_model_class():
    from .gptoss_native_model import NativeModel

    class CompilerSafeNativeModel(NativeModel):
        def __init__(self, *args, **kwargs):
            repair = install()
            super().__init__(*args, **kwargs)
            self.metadata.update(compiler_capacity_repair=repair,
                inherited_compute_unchanged=False,
                compute_amendment='Compiler capacity and fail-closed policy only; same mathematical attention')

    return CompilerSafeNativeModel
