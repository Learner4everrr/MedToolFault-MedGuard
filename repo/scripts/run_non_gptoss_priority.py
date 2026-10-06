"""Persistent other-model continuation; no deferred-model auto-resume."""
import argparse
import os
import time
import traceback
from pathlib import Path

from medtoolbench.supplement.non_gptoss_priority import prepare, run
from medtoolbench.supplement.storage_recovery import atomic_json

p = argparse.ArgumentParser()
p.add_argument('--unit', type=Path, required=True)
a = p.parse_args()
control = Path('data/non_gptoss_priority_v1').resolve()
control.mkdir(parents=True, exist_ok=True)
log = (control / 'bootstrap.log').open('x')
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
try:
    prepare(control, a.unit.resolve())
    atomic_json(control / 'bootstrap.json', {'state': 'supervising_other_models', 'time': time.time()})
    run(control / 'manifest.json')
except BaseException:
    atomic_json(control / 'bootstrap.json', {'state': 'failed', 'error': traceback.format_exc(), 'time': time.time()})
    raise
