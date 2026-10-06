"""Fail-closed persistent launch after exact-source synthetic roundtrip proof."""
import argparse
import os
import time
import traceback
from pathlib import Path

from medtoolbench.supplement.gptoss_protocol_pipeline import prepare, run
from medtoolbench.supplement.storage_recovery import atomic_json, read

p = argparse.ArgumentParser()
p.add_argument('--roundtrip', type=Path, required=True)
p.add_argument('--unit', type=Path, required=True)
a = p.parse_args()
control = Path('data/gptoss_protocol_repair_v2').resolve()
log = (control / 'bootstrap.log').open('x')
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
status = control / 'bootstrap.json'
deadline = time.monotonic() + 7200
try:
    while not a.roundtrip.exists() or not a.unit.exists():
        failure = read(a.roundtrip.parent / 'failure.json')
        if failure:
            raise RuntimeError('Synthetic integration failed; no automatic experiment launch: ' + str(failure))
        if time.monotonic() > deadline:
            raise RuntimeError('Validation waiting deadline exceeded')
        atomic_json(status, {'state': 'waiting_for_validation', 'time': time.time()})
        time.sleep(10)
    prepare(control, a.roundtrip.resolve(), a.unit.resolve())
    atomic_json(status, {'state': 'supervising_queue', 'time': time.time()})
    run(control / 'manifest.json')
except BaseException:
    atomic_json(status, {'state': 'failed', 'error': traceback.format_exc(), 'time': time.time()})
    raise
