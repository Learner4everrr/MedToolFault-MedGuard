"""One physical inference item at a time; keep logical episode batches intact.

Seed values, task order, precision, context and output budgets remain unchanged.
Sampling RNG consumption and floating-point rounding can differ from batched
generation. This is an explicit execution stratum, NOT a bitwise equivalence
claim. Never dynamically choose this policy based on an individual outcome.
"""
from .memory_adapter import install
from .runner import SampleModel


class SerialGemma(SampleModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.kind != "gemma3":
            raise RuntimeError("Serial adapter is scoped to Gemma")
        adapter = install(self.model, 512)
        self.metadata["serial_execution"] = {**adapter, "physical_inference_batch_size": 1,
            "logical_episode_batch_unchanged": True, "stochastic_rng_equivalence_claimed": False,
            "universal_bitwise_equivalence_claimed": False,
            "wall_time_pooling_with_original_execution_allowed": False}

    def generate(self, prompts, max_new_tokens):
        generate = super().generate
        return [row for prompt in prompts
                for row in generate([prompt], max_new_tokens)]

    def generate_sample(self, prompts, limit, temperature):
        if not temperature:
            return self.generate(prompts, limit)
        generate_sample = super().generate_sample
        return [row for prompt in prompts
                for row in generate_sample([prompt], limit, temperature)]
