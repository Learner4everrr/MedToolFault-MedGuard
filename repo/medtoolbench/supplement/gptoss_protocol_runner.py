"""New transport revision around unchanged clinical experiment runners."""
import argparse
import importlib
from pathlib import Path

from . import runner
from .gptoss_protocol_model import GptOssProtocolModel
from .input_guard import install
from .isolated_execution import model_gate
from .storage_recovery import read


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args, _ = parser.parse_known_args()
    plan = runner.load_plan(args.plan)
    job = next(j for j in plan['jobs'] if j['id'] == args.job)
    if job['model'] != 'gptoss20b' or {s['name'] for s in plan['systems']} != {'gptoss20b'}:
        raise RuntimeError('GPT-OSS only; never dispatch Gemma')
    amendment = plan['gptoss_protocol_repair']
    proof = read(amendment['roundtrip_proof'])
    if not proof or not proof.get('all_pass') or proof.get('held_out_tasks_used') is not False:
        raise RuntimeError('Exact-source synthetic roundtrip proof required')
    if job['stage'] == 'test' and model_gate(args.plan, plan, 'gptoss20b') != 'passed':
        raise RuntimeError('Fresh development audit required; never gate on accuracy')
    mode = 'external' if plan['runner_module'].endswith('external_runner') else 'fhir'
    guard = install()
    record = {'amendment': amendment, 'plan_hash': plan['plan_hash'], 'input_guard': guard,
        'clinical_prompts_labels_budgets_grader_unchanged': True, 'inference_api': False}

    class RecordedModel(GptOssProtocolModel):
        protocol_mode = mode

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.metadata['gptoss_protocol_repair'] = record

    runner.SampleModel = RecordedModel
    module = importlib.import_module(plan['runner_module'])
    if mode == 'external':
        module.SampleModel = RecordedModel
    original = runner.write_json

    def write(path, value):
        if Path(path).name == 'config.json':
            value = {**value, 'gptoss_protocol_repair': record}
        return original(path, value)

    runner.write_json = write
    if mode == 'external':
        module.write_json = write
    module.main()


if __name__ == '__main__':
    main()
