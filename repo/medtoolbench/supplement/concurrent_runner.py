"""Install the separately certified transport revision on original job inputs."""
import argparse
import sys
from pathlib import Path


def main():
    from .concurrent_pipeline import load, repair_key
    from .concurrent_repair import ConcurrentRejectionProxy, REVISION
    from . import runner, corrected_runner, isolated_runner
    from .runner import load_plan
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--repair-manifest', type=Path, required=True)
    a, remaining = p.parse_known_args()
    manifest = load(a.repair_manifest.resolve())
    cp = Path(remaining[remaining.index('--plan') + 1]).resolve()
    job_id = remaining[remaining.index('--job') + 1]
    if repair_key(cp, job_id) not in manifest['repair_jobs']:
        raise RuntimeError('Job is outside reviewed repair scope')
    plan = load_plan(cp)
    if plan['runner_module'] != 'medtoolbench.supplement.corrected_runner':
        raise RuntimeError('Only the declared A transport runner may be repaired')
    corrected_runner.InterfaceConsistentProxy = ConcurrentRejectionProxy
    original_write = runner.write_json

    def recorded_write(path, value):
        if Path(path).name == 'config.json':
            value = {**value, 'concurrent_repair': {'revision': REVISION,
                'manifest': str(a.repair_manifest.resolve()), 'manifest_hash': manifest['manifest_hash'],
                'source_hashes': manifest['repair_source_hashes'], 'payload_corrected': False,
                'scientific_plan_hash': plan['plan_hash'], 'budgets_prompts_grader_unchanged': True}}
        return original_write(path, value)

    runner.write_json = recorded_write
    sys.argv = [isolated_runner.__file__, '--execution-manifest', manifest['execution_manifest'], *remaining]
    isolated_runner.main()


if __name__ == '__main__':
    main()
