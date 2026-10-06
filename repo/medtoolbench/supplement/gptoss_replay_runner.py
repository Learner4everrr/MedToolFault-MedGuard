"""Unchanged native inference with a versioned post-save audit replacement."""
import argparse
import sys
import time
import traceback
from pathlib import Path

from . import gptoss_native_pipeline as native
from . import gptoss_native_runner as entry
from .gptoss_replay_audit import audit_native
from .gptoss_replay_recovery import load, development_passed, authorize_job
from .runner import load_plan
from .storage_recovery import atomic_json


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--replay-manifest', type=Path, required=True)
    a, rest = p.parse_known_args()
    cp = Path(rest[rest.index('--plan') + 1]).resolve()
    out = Path(rest[rest.index('--out') + 1]).resolve()
    manifest = load(a.replay_manifest)
    plan = load_plan(cp)
    job = next(j for j in plan['jobs'] if j['id'] == rest[rest.index('--job') + 1])
    if out != cp.parent / 'jobs' / job['id'] / 'attempt1':
        raise RuntimeError('Unexpected recovery output path')
    authorize_job(manifest, cp, plan, job, allow_launch_log=True)
    native.load, native.development_passed = load, development_passed
    entry.audit_native = audit_native
    sys.argv = [entry.__file__, '--native-manifest', str(a.replay_manifest), *rest]
    try:
        entry.main()
    except BaseException:
        if out.exists():
            atomic_json(out / 'native_audit_failure.json', {'error': traceback.format_exc(),
                'manifest_hash': manifest['manifest_hash'], 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
