"""Record the Llama amendment while retaining the certified clinical runners."""
import argparse
import importlib
import sys
from pathlib import Path

from .acd_expansion import MODEL, load
from .input_guard import install as install_input_guard
from .isolated_execution import model_gate
from .runner import load_plan


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--expansion-manifest', type=Path, required=True)
    args, remaining = p.parse_known_args()
    path = args.expansion_manifest.resolve()
    manifest = load(path, require_handoff=True)
    cp = Path(remaining[remaining.index('--plan') + 1]).resolve()
    if str(cp) not in manifest['new_campaigns']:
        raise RuntimeError('Plan outside full-Llama expansion scope')
    plan = load_plan(cp)
    if plan['plan_hash'] != manifest['plan_hashes'][str(cp)]:
        raise RuntimeError('Expansion plan identity mismatch')
    job = next(j for j in plan['jobs'] if j['id'] == remaining[remaining.index('--job') + 1])
    if job['model'] != MODEL:
        raise RuntimeError('Only the requested Llama model can enter this runner')
    if job['stage'] == 'test' and model_gate(cp, plan, MODEL) != 'passed':
        raise RuntimeError('Fresh development qualification must precede formal expansion')
    from . import runner, corrected_runner
    from medtoolbench.research import run as original_run
    from .concurrent_repair import ConcurrentRejectionProxy, REVISION
    original_parser = original_run.parse_action
    guard = install_input_guard()
    concurrent = bool(job['settings'].get('stress', {}).get('concurrent'))
    record = {'manifest': str(path), 'manifest_hash': manifest['manifest_hash'],
        'source_hashes': manifest['source_hashes'], 'scientific_plan_hash': plan['plan_hash'],
        'input_guard': guard, 'concurrent_repair': REVISION if concurrent else None,
        'payload_corrected': False, 'original_budgets_prompts_grader_retained': True,
        'json_envelope_candidate_enabled': False, 'inference_api': False}
    parent = runner.SampleModel

    class RecordedModel(parent):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.metadata['acd_expansion'] = record

    original_write = runner.write_json
    original_proxy = corrected_runner.InterfaceConsistentProxy
    original_stress, original_hapi = runner.StressProxy, runner.HapiProcess
    original_argv = sys.argv

    def recorded_write(file, value):
        if Path(file).name == 'config.json':
            value = {**value, 'acd_expansion': record}
        return original_write(file, value)

    try:
        runner.SampleModel = RecordedModel
        runner.write_json = recorded_write
        if concurrent:
            if not plan['runner_module'].endswith('corrected_runner'):
                raise RuntimeError('Unexpected concurrent campaign runner')
            corrected_runner.InterfaceConsistentProxy = ConcurrentRejectionProxy
        module = importlib.import_module(plan['runner_module'])
        sys.argv = [module.__file__, *remaining]
        module.main()
    finally:
        runner.SampleModel, runner.write_json = parent, original_write
        corrected_runner.InterfaceConsistentProxy = original_proxy
        runner.StressProxy, runner.HapiProcess = original_stress, original_hapi
        original_run.parse_action = original_parser
        sys.argv = original_argv


if __name__ == '__main__':
    main()
