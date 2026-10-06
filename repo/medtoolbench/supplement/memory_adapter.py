"""Inference-only Gemma FFN tiling; no attention/cache/token/batch changes."""
from types import MethodType


def tiled_forward(self, x):
    import torch
    if torch.is_grad_enabled() or self.training:
        raise RuntimeError("FFN memory adapter is inference-only")
    size = self._memory_chunk_tokens
    if x.ndim != 3 or x.shape[1] <= size:
        return self._memory_original_forward(x)
    # The FFN has no interaction between token positions. Keep attention and its
    # entire context untouched; materialize only a small intermediate at a time.
    output = torch.empty_like(x)
    for offset in range(0, x.shape[1], size):
        output[:, offset:offset + size, :] = self._memory_original_forward(x[:, offset:offset + size, :])
    return output


def install(model, chunk_tokens=512):
    from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP
    if not isinstance(chunk_tokens, int) or chunk_tokens < 1:
        raise ValueError("Positive integer chunk size required")
    modules = [m for m in model.modules() if isinstance(m, Gemma3MLP)]
    if not modules:
        raise RuntimeError("Expected Gemma3 FFN modules")
    for module in modules:
        if hasattr(module, "_memory_original_forward"):
            raise RuntimeError("Adapter already installed")
    for module in modules:
        module._memory_original_forward = module.forward
        module._memory_chunk_tokens = chunk_tokens
        module.forward = MethodType(tiled_forward, module)
    return {"adapter": "gemma3_tokenwise_ffn_tiling_v1", "chunk_tokens": chunk_tokens,
            "layers": len(modules), "batch_size_changed": False, "attention_changed": False,
            "context_or_generation_limit_changed": False, "dtype_changed": False,
            "numerical_note": "Mathematically tokenwise-equivalent; floating-point kernel shapes can differ. See preflight comparisons."}


def uninstall(model):
    for module in model.modules():
        if hasattr(module, "_memory_original_forward"):
            module.forward = module._memory_original_forward
            del module._memory_original_forward
            del module._memory_chunk_tokens
