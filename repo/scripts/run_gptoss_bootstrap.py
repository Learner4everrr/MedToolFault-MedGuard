"""Persistent, fail-closed handoff from GPU preflight into the experiment queue."""
import argparse
import os
import time
import traceback
from pathlib import Path
from medtoolbench.supplement.gptoss_pipeline import prepare, run
from medtoolbench.supplement.storage_recovery import atomic_json, read

p = argparse.ArgumentParser()
p.add_argument('--smoke', type=Path, required=True)
p.add_argument('--unit-proof', type=Path, required=True)
a = p.parse_args()
control = Path('data/gptoss_replacement_v1').resolve()
name = 'bootstrap_' + a.smoke.parent.name
log = (control / (name + '.log')).open('x')
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
status = control / (name + '.json')
deadline = time.monotonic() + 7200
mask_proof = control / 'mask_equivalence_v1/complete.json'
required = [a.smoke, a.unit_proof, mask_proof]
while not all(path.exists() for path in required):
    failure = read(a.smoke.parent / 'failure.json') or read(mask_proof.parent / 'failure.json')
    if failure:
        atomic_json(status, {'state': 'preflight_failed', 'failure': failure, 'time': time.time()})
        raise RuntimeError('GPU preflight failed; formal dispatch prohibited')
    if time.monotonic() > deadline:
        raise RuntimeError('Preflight waiting deadline exceeded')
    atomic_json(status, {'state': 'waiting_for_validation', 'smoke': str(a.smoke),
        'pending': [str(path) for path in required if not path.exists()], 'time': time.time()})
    time.sleep(15)
atomic_json(status, {'state': 'freezing_and_launching', 'time': time.time()})
try:
    prepare(control, a.smoke.resolve(), a.unit_proof.resolve())
    atomic_json(status, {'state': 'supervising_queue', 'time': time.time()})
    run(control / 'manifest.json')
except BaseException:
    atomic_json(status, {'state': 'launch_or_controller_failed', 'error': traceback.format_exc(), 'time': time.time()})
    raise
