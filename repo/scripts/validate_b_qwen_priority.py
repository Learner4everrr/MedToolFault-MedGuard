"""CPU regression evidence for the Qwen-only B scheduling amendment."""
import argparse
import time
import unittest
from pathlib import Path

from medtoolbench.supplement.b_qwen_priority import ROOT, source_hashes
from medtoolbench.supplement.storage_recovery import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise RuntimeError('Never overwrite validation evidence')
    before = source_hashes()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'repo/tests'), pattern='test*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or source_hashes() != before:
        raise RuntimeError('Full exact-source regression required')
    atomic_json(args.out, {'all_pass': True, 'tests': result.testsRun, 'skipped': result.skipped,
        'source_hashes': before, 'held_out_tasks_used': False, 'time': time.time()})


if __name__ == '__main__':
    main()
