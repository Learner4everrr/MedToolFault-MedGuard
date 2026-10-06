"""GPT-OSS-only adapter around unchanged A/B/C/D scientific runners."""
import argparse
import importlib
import json
import sys
from pathlib import Path

from medtoolbench.storage import digest
from . import runner
from .gptoss_model import GptOssModel
from .input_guard import install
from .isolated_execution import model_gate


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--job', required=True)
    a, _ = p.parse_known_args()
    plan = runner.load_plan(a.plan)
    job = next(j for j in plan['jobs'] if j['id'] == a.job)
    if job['model'] != 'gptoss20b' or {s['name'] for s in plan['systems']} != {'gptoss20b'}:
        raise RuntimeError('This runner cannot execute Gemma or any other model')
    amendment = plan['gptoss_replacement']
    proof = json.loads(Path(amendment['smoke']).read_text())
    if not proof['all_pass'] or proof['adapter_hash'] != digest(Path(__file__).with_name('gptoss_model.py').read_text()):
        raise RuntimeError('Require a passing smoke for this exact adapter')
    if job['stage'] == 'test' and model_gate(a.plan, plan, 'gptoss20b') != 'passed':
        raise RuntimeError('Development audit gate not yet passed; never gate on accuracy')
    guard = install()
    record = {'amendment': amendment, 'plan_hash': plan['plan_hash'], 'input_guard': guard,
        'clinical_prompts_labels_budgets_grader_unchanged': True, 'inference_api': False}
    class RecordedModel(GptOssModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.metadata['gptoss_replacement'] = record
    runner.SampleModel = RecordedModel
    module = importlib.import_module(plan['runner_module'])
    if plan['runner_module'].endswith('external_runner'):
        module.SampleModel = RecordedModel
    original = runner.write_json
    def write(path, value):
        if Path(path).name == 'config.json':
            value = {**value, 'gptoss_replacement': record}
        return original(path, value)
    runner.write_json = write
    if plan['runner_module'].endswith('external_runner'):
        module.write_json = write
    module.main()


if __name__ == '__main__':
    main()
