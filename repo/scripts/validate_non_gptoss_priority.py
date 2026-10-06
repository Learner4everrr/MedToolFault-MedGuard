"""CPU regression receipt for the other-model-only scheduling change."""
import argparse
import time
import unittest
from pathlib import Path

from medtoolbench.supplement.non_gptoss_priority import source_hashes
from medtoolbench.supplement.storage_recovery import atomic_json

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if a.out.exists():
    raise RuntimeError('Never overwrite validation')
root = Path(__file__).resolve().parents[2]
sources = source_hashes(root)
suite = unittest.defaultTestLoader.discover(str(root / 'repo/tests'), pattern='test*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped or source_hashes(root) != sources:
    raise RuntimeError('Exact-source full regression required')
atomic_json(a.out, {'all_pass': True, 'tests': result.testsRun, 'source_hashes': sources,
    'held_out_tasks_used': False, 'time': time.time()})
