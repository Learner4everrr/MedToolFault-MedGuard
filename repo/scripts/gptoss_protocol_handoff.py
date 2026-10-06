"""Preserve evidence and stop only the verified v1 GPT-OSS protocol workers."""
import json
import os
import signal
import time
from pathlib import Path

from medtoolbench.supplement.gptoss_pipeline import process_cmdline
from medtoolbench.supplement.storage_recovery import atomic_json, read

root = Path(__file__).resolve().parents[2]
old = root / 'data/gptoss_replacement_v1'
control = root / 'data/gptoss_protocol_repair_v2'
target = control / 'handoff.json'
if target.exists():
    raise RuntimeError('Never overwrite a handoff')
status = read(old / 'status.json')
pid = status['pid']
cmdline = process_cmdline(pid)
if 'run_gptoss_bootstrap.py' not in cmdline:
    raise RuntimeError('Controller identity changed')
os.kill(pid, signal.SIGSTOP)
workers = []
for path in sorted((old / 'workers').glob('*.json')):
    row = read(path)
    current = process_cmdline(row['pid'])
    if not current:
        continue
    if row['out'] not in current or '-m medtoolbench.supplement.gptoss_runner ' not in current:
        raise RuntimeError('Unexpected worker; do not signal it')
    row['observed_cmdline'] = current
    row['config'] = read(Path(row['out']) / 'config.json')
    if row['config']['job']['model'] != 'gptoss20b':
        raise RuntimeError('Must not interrupt another model')
    workers.append(row)
inventory = []
for campaign in sorted(old.glob('campaign_*')):
    for job in sorted((campaign / 'jobs').glob('*')):
        for attempt in sorted(job.glob('attempt[0-9]*')):
            if not attempt.is_dir():
                continue
            config = read(attempt / 'config.json')
            if not config or config['job']['stage'] != 'qualification':
                raise RuntimeError('Unexpected formal evidence: inspect before proceeding')
            inventory.append({'run': str(attempt), 'job': config['job']['id'],
                'disposition': 'adapter-invalid development; excluded from all v2 gates and scientific estimates'})
preserved = []
for row in read(old / 'handoff.json')['live_children_preserved']:
    current = process_cmdline(row['pid'])
    if current:
        if row['out'] not in current:
            raise RuntimeError('Preserved PID reused')
        preserved.append(row)
atomic_json(target, {'time': time.time(), 'old_controller_pid': pid,
    'old_controller_cmdline': cmdline, 'old_status': status,
    'live_children_preserved': preserved, 'invalidated_children_stopped': workers,
    'invalidated_development': inventory, 'formal_gptoss_attempts_observed': 0,
    'reason': 'v1 drops valid Harmony tool envelopes before unchanged action dispatch; no clinical result used for repair',
    'old_control': str(old), 'inference_api': False})
for row in workers:
    if process_cmdline(row['pid']) != row['observed_cmdline']:
        raise RuntimeError('Identity changed after snapshot')
    os.kill(row['pid'], signal.SIGTERM)
# SIGTERM is delivered before a stopped process resumes user-space dispatch.
os.kill(pid, signal.SIGTERM)
os.kill(pid, signal.SIGCONT)
atomic_json(old / 'superseded_protocol.json', {'time': time.time(),
    'successor': str(control), 'snapshot': str(target), 'all_gptoss_development_invalidated': True,
    'historical_evidence_retained': True, 'other_model_results_unaffected': True})
print(json.dumps({'stopped_gptoss': [r['pid'] for r in workers],
    'preserved_other_models': [r['pid'] for r in preserved], 'formal_attempts': 0}), flush=True)
