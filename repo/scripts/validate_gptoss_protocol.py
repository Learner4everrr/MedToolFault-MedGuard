"""Freeze a full CPU-regression receipt for the exact protocol-repair sources."""
import argparse
import time
import unittest
from pathlib import Path

from medtoolbench.supplement.gptoss_protocol_pipeline import source_hashes
from medtoolbench.supplement.storage_recovery import atomic_json

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if a.out.exists():
    raise RuntimeError('Never overwrite a validation receipt')
root = Path(__file__).resolve().parents[2]
before = source_hashes(root)
suite = unittest.defaultTestLoader.discover(str(root / 'repo/tests'), pattern='test*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped or source_hashes(root) != before:
    raise RuntimeError('Complete exact-source regression validation required')
atomic_json(a.out, {'all_pass': True, 'tests': result.testsRun, 'source_hashes': before,
    'skipped': result.skipped, 'held_out_tasks_used': False, 'time': time.time()})
