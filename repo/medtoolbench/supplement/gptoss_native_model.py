"""Native conversation adapter around the already validated offline backend."""
from medtoolbench.storage import digest
from .gptoss_model import GptOssModel
from .gptoss_native import VERSION, INSTRUCTION, TOOLS, decode, render


class NativeModel(GptOssModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata.update(protocol=VERSION, transport_instruction=INSTRUCTION, tools=TOOLS,
            native_history=True, argument_repair=False, clinical_base_prompt_preserved=True,
            prompt_serialization_changed=True, inherited_compute_unchanged=True)

    def prompt(self, messages):
        return render(self.tokenizer, self.template, messages)

    def generate_sample(self, prompts, limit, temperature):
        rows = super().generate_sample(prompts, limit, temperature)
        for prompt, row in zip(prompts, rows):
            if 'native_transport' not in row:
                row['legacy_harmony_status'] = row['harmony_status']
                row['text'], row['harmony_status'], row['native_transport'] = decode(row['raw_harmony'])
            row['native_prompt_hash'] = digest(prompt)
        return rows
