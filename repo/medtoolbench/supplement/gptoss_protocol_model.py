"""Post-decode-only adapter; inherits all validated native MXFP4 compute."""
from .gptoss_model import GptOssModel
from .gptoss_protocol import VERSION, decode


class GptOssProtocolModel(GptOssModel):
    protocol_mode = 'fhir'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata['protocol'] = VERSION
        self.metadata['protocol_mode'] = self.protocol_mode
        self.metadata['compute_and_prompt_inherited_unchanged'] = True

    def generate_sample(self, prompts, limit, temperature):
        rows = super().generate_sample(prompts, limit, temperature)
        for row in rows:
            if 'protocol_bridge' in row:  # base class recursively microbatches
                continue
            row['legacy_harmony_status'] = row['harmony_status']
            row['text'], row['harmony_status'], row['protocol_bridge'] = decode(
                row['raw_harmony'], self.protocol_mode)
        return rows
