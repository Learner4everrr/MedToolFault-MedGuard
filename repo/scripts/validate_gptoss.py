"""Run the existing CPU regression suite plus migration tests; freeze proof."""
import sys
import time
import unittest
import argparse
import json
from pathlib import Path
from medtoolbench.supplement.gptoss_pipeline import source_hashes
from medtoolbench.supplement.storage_recovery import atomic_json

root = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, default=root / 'data/gptoss_replacement_v1/unit_tests_final.json')
p.add_argument('--migration-only', action='store_true')
p.add_argument('--regression-ref', type=Path)
args = p.parse_args()
out = args.out
if out.exists():
    raise RuntimeError('Validation receipt already exists; do not overwrite')
before = source_hashes(root)
regression = None
if args.migration_only:
    if not args.regression_ref:
        raise RuntimeError('Focused validation requires the completed full regression receipt')
    regression = json.loads(args.regression_ref.read_text())
    if not regression.get('all_pass') or regression.get('tests', 0) < 174:
        raise RuntimeError('Invalid full-regression reference')
suite = unittest.defaultTestLoader.discover(str(root / 'repo/tests'),
    pattern='test_gptoss_replacement.py' if args.migration_only else 'test*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    sys.exit(1)
if source_hashes(root) != before:
    raise RuntimeError('Sources changed while validation ran; use a fresh validation receipt')
atomic_json(out, {'all_pass': True, 'tests': result.testsRun, 'skipped': result.skipped,
    'source_hashes': before, 'time': time.time(), 'held_out_tasks_used': False,
    'full_regression_reference': str(args.regression_ref) if regression else None,
    'full_regression_tests': regression['tests'] if regression else result.testsRun})
