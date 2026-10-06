"""Full exact-source CPU suite before the certified repair queue takes ownership."""
import argparse
import time
import unittest
from pathlib import Path

from medtoolbench.supplement.concurrent_pipeline import ROOT, source_hashes
from medtoolbench.supplement.storage_recovery import atomic_json

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if a.out.exists():
    raise RuntimeError('Never overwrite validation')
sources = source_hashes()
suite = unittest.defaultTestLoader.discover(str(ROOT / 'repo/tests'))
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped or source_hashes() != sources:
    raise RuntimeError('Exact-source full regression required')
atomic_json(a.out, {'all_pass': True, 'tests': result.testsRun, 'source_hashes': sources,
    'held_out_tasks_used': False, 'time': time.time()})
