"""Stop verified v2 GPT-OSS workers, retain evidence and other-model ownership."""
import os
import signal
import time
from pathlib import Path

from medtoolbench.supplement.gptoss_pipeline import process_cmdline
from medtoolbench.supplement.storage_recovery import atomic_json, read

root = Path(__file__).resolve().parents[2]
old = root / 'data/gptoss_protocol_repair_v2'
control = root / 'data/gptoss_native_v3'
out = control / 'handoff.json'
if out.exists():
    raise RuntimeError('Never overwrite handoff evidence')
status = read(old / 'status.json')
pid = status['pid']
cmd = process_cmdline(pid)
if 'repo/scripts/run_gptoss_protocol.py' not in cmd:
    raise RuntimeError('Dispatcher identity changed')
os.kill(pid, signal.SIGSTOP)
stopped, preserved = [], []
for path in sorted((old / 'workers').glob('*.json')):
    row = read(path)
    current = process_cmdline(row['pid'])
    if not current:
        continue
    if row['out'] not in current:
        raise RuntimeError('Worker PID reused')
    row['observed_cmdline'] = current
    row['config'] = read(Path(row['out']) / 'config.json')
    if row['config']['job']['model'] == 'gptoss20b':
        if '-m medtoolbench.supplement.gptoss_protocol_runner ' not in current:
            raise RuntimeError('Unexpected GPT-OSS process identity')
        stopped.append(row)
    else:
        preserved.append(row)
for row in read(old / 'handoff.json')['live_children_preserved']:
    current = process_cmdline(row['pid'])
    if current and row['out'] not in current:
        raise RuntimeError('Preserved PID reused')
    # Include dead-but-unadopted children too, so their evidence is reconciled.
    preserved.append(row)
inventory = []
for path in sorted(old.glob('campaign_*/jobs/*/attempt[0-9]*/config.json')):
    config = read(path)
    inventory.append({'run': str(path.parent), 'job': config['job']['id'],
        'stage': config['job']['stage'], 'plan_hash': config['plan_hash'],
        'completed_before_pause': (path.parent / 'complete.json').exists(),
        'disposition': 'interface-confounded historical cohort; not pooled with v3'})
atomic_json(out, {'time': time.time(), 'old_controller_pid': pid, 'old_controller_cmdline': cmd,
    'old_status': status, 'live_children_preserved': preserved, 'gptoss_children_stopped': stopped,
    'retained_v2_attempts': inventory, 'old_control': str(old), 'inference_api': False,
    'reason': 'User requests pause, native protocol repair, validation, then continuation. No held-out outcomes used for adaptation.'})
for row in stopped:
    if process_cmdline(row['pid']) != row['observed_cmdline']:
        raise RuntimeError('Worker identity changed before signal')
    os.kill(row['pid'], signal.SIGTERM)
os.kill(pid, signal.SIGTERM)
os.kill(pid, signal.SIGCONT)
atomic_json(old / 'paused_native_repair.json', {'time': time.time(), 'successor': str(control),
    'snapshot': str(out), 'preserve_all_evidence': True, 'do_not_restart_v2': True})
print({'stopped_gptoss': [r['pid'] for r in stopped], 'preserved': [r['pid'] for r in preserved],
       'retained_attempts': len(inventory)}, flush=True)
