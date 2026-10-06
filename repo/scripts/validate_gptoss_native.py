"""Exact-source full regression receipt, with no benchmark outcome selection."""
import argparse
import time
import unittest
from pathlib import Path

from medtoolbench.supplement.gptoss_native_pipeline import source_hashes, ROOT
from medtoolbench.supplement.storage_recovery import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('Never overwrite a validation receipt')
    before = source_hashes()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'repo/tests'), pattern='test*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or source_hashes() != before:
        raise RuntimeError('Complete exact-source regression validation required')
    atomic_json(a.out, {'all_pass': True, 'tests': result.testsRun, 'source_hashes': before,
        'skipped': result.skipped, 'held_out_tasks_used': False, 'time': time.time()})


if __name__ == '__main__':
    main()
