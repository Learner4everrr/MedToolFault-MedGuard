"""Exact-source full regression certificate for the audit-only repair."""
import argparse
import time
import unittest
from pathlib import Path

from medtoolbench.supplement.gptoss_replay_recovery import ROOT, source_hashes
from medtoolbench.supplement.gptoss_native_pipeline import source_hashes as native_sources
from medtoolbench.supplement.storage_recovery import atomic_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('Never overwrite validation evidence')
    before, original = source_hashes(), native_sources()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'repo/tests'), pattern='test*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or source_hashes() != before or native_sources() != original:
        raise RuntimeError('Complete exact-source regression required')
    atomic_json(a.out, {'all_pass': True, 'tests': result.testsRun, 'skipped': result.skipped,
        'source_hashes': before, 'native_source_hashes': original, 'held_out_tasks_used': False, 'time': time.time()})


if __name__ == '__main__':
    main()
