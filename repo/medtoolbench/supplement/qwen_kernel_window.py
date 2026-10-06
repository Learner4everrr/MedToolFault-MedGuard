"""Bounded development window: pause dispatch only, never interrupt formal work.

The original dispatcher is SIGSTOP'ed only near a natural shard boundary.
Independent formal children continue. Resume the same dispatcher after releasing
the diagnostic GPU. A separate watchdog restores it if this driver dies.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from medtoolbench.research.cluster_execution import FileLease
from . import b_qwen_priority as original
from . import qwen_kernel_probe as probe
from .memory_execution import verify_completion
from .storage_recovery import atomic_json, read


def identity(pid):
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().split(') ', 1)[1].split()
        return {'pid': pid, 'start_ticks': fields[19], 'state': fields[0],
            'cmdline': (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()}
    except FileNotFoundError:
        return None


def same_process(expected):
    current = identity(expected['pid'])
    return bool(current and current['start_ticks'] == expected['start_ticks']
                and current['cmdline'] == expected['cmdline'] and current['state'] != 'Z')


def resume(control, parent):
    if same_process(parent):
        os.kill(parent['pid'], signal.SIGCONT)
    atomic_json(control / 'dispatch_resumed.json', {'time': time.time(), 'parent': parent,
        'parent_alive': same_process(parent)})


def stop_probe(control):
    record = read(control / 'active_probe.json')
    if not record or not same_process(record):
        return
    if '-m medtoolbench.supplement.qwen_kernel_probe run ' not in record['cmdline']:
        raise RuntimeError('Unexpected diagnostic process; refuse signal')
    os.killpg(record['pid'], signal.SIGTERM)
    for _ in range(50):
        if not same_process(record):
            return
        time.sleep(.1)
    if same_process(record):
        os.killpg(record['pid'], signal.SIGKILL)
        for _ in range(100):
            if not same_process(record):
                return
            time.sleep(.1)
        raise RuntimeError('Probe still owns GPU; original dispatcher must stay paused')


def watchdog(control):
    record = read(control / 'window.json')
    while not (control / 'dispatch_resumed.json').exists():
        if not same_process(record['driver']) or time.time() > record['deadline']:
            stop_probe(control)
            resume(control, record['parent'])
            atomic_json(control / 'watchdog_recovery.json', {'time': time.time(), 'reason': 'Driver exited or bounded window expired'})
            return
        time.sleep(5)


def near_boundary():
    for file in sorted((original.CONTROL / 'workers').glob('*.json')):
        r = read(file)
        if not r or r.get('state') != 'running':
            continue
        p = read(Path(r['out']) / 'progress.json')
        if p and 0 <= p['total']-p['completed'] <= 4:
            return True
    return False


def capture_children(plan):
    rows = []
    jobs = {j['id']: j for j in original.selected(plan, 'test')}
    for file in sorted((original.CONTROL / 'workers').glob('*.json')):
        r = read(file)
        if not r or r.get('state') != 'running':
            continue
        out = Path(r['out'])
        cfg = read(out / 'config.json')
        if not cfg or cfg['job'] != jobs.get(cfg['job']['id']) or cfg['plan_hash'] != plan['plan_hash']:
            raise RuntimeError('Formal child assignment changed')
        current = identity(r['pid'])
        if current and current['state'] != 'Z' and (str(out) not in current['cmdline'] or
                '-m medtoolbench.supplement.isolated_runner ' not in current['cmdline']):
            raise RuntimeError('Unexpected formal child identity')
        rows.append({'worker': r['worker'], 'pid': r['pid'], 'identity': current,
                     'out': str(out), 'job': cfg['job']})
    if not rows:
        raise RuntimeError('No formal children to drain')
    return rows


def validate_unit(control):
    u = read(control / 'unit.json')
    if not u or not u.get('all_pass') or u.get('source_hashes') != probe.source_hashes() or u.get('skipped'):
        raise RuntimeError('Exact-source regression proof required')


def run(control):
    m, plan, _ = probe.load(control)
    validate_unit(control)
    parent = identity(read(original.CONTROL / 'started.json')['pid'])
    if not parent or '-m medtoolbench.supplement.b_qwen_priority ' not in parent['cmdline'] or parent['state'] in ('T', 't', 'Z'):
        raise RuntimeError('Original dispatcher is not the expected running process')
    with FileLease(control / 'driver.lock'):
        if (control / 'driver_started.json').exists():
            raise RuntimeError('Never restart a diagnostic window without reconciliation')
        atomic_json(control / 'driver_started.json', {'time': time.time(), 'pid': os.getpid(),
            'manifest_hash': m['manifest_hash']})
        wait_until = time.time()+7200
        while not near_boundary():
            if not same_process(parent) or time.time() > wait_until:
                raise RuntimeError('No safe natural shard boundary within two hours')
            atomic_json(control / 'status.json', {'state': 'waiting_natural_boundary', 'time': time.time(),
                'formal_dispatch_running': True})
            time.sleep(5)
        driver = identity(os.getpid())
        window = {'parent': parent, 'driver': driver, 'deadline': time.time()+7200,
                  'time': time.time(), 'formal_children_signalled': False}
        atomic_json(control / 'window.json', window)
        with (control / 'watchdog.log').open('x') as log:
            subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, 'watchdog', '--control', str(control)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        paused = False
        try:
            if not same_process(parent):
                raise RuntimeError('Dispatcher identity changed before pause')
            os.kill(parent['pid'], signal.SIGSTOP)
            paused = True
            # Wait for all dispatcher threads to stop before recording workers.
            for _ in range(50):
                current = identity(parent['pid'])
                if current and current['state'] == 'T':
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('Dispatcher did not pause')
            children = capture_children(plan)
            atomic_json(control / 'preserved_children.json', children)
            atomic_json(original.CONTROL / 'kernel_probe_window.json', {
                'control': str(control), 'time': time.time(), 'formal_children_preserved': True,
                'resume_receipt': str(control / 'dispatch_resumed.json')})
            lane = None
            while lane is None:
                for row in children:
                    current = identity(row['pid'])
                    if current and current['state'] != 'Z':
                        continue
                    if (Path(row['out']) / 'complete.json').exists():
                        verify_completion(Path(row['out']), plan, row['job'])
                        lane = row
                        break
                    raise RuntimeError('A formal child exited without a valid completion; resume original supervisor for diagnosis')
                if time.time() > window['deadline'] or (control / 'dispatch_resumed.json').exists():
                    raise RuntimeError('Diagnostic window expired')
                atomic_json(control / 'status.json', {'state': 'draining_one_lane', 'time': time.time(),
                    'formal_children_preserved': True, 'formal_dispatch_paused': True})
                if lane is None:
                    time.sleep(5)
            atomic_json(control / 'lane.json', lane)
            for backend in ('fast', 'reference'):
                if (control / 'dispatch_resumed.json').exists():
                    raise RuntimeError('Cannot start probe after dispatcher resumed')
                python = m['fast_python'] if backend == 'fast' else m['reference_python']
                cmd = [python, '-u', '-m', 'medtoolbench.supplement.qwen_kernel_probe', 'run',
                    '--control', str(control), '--backend', backend, '--worker', lane['worker']['name']]
                with (control / (backend+'.log')).open('x') as log:
                    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(lane['worker']['gpu']),
                                 HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
                    child_id = identity(child.pid)
                    # exec may not yet have replaced the fork's command line.
                    for _ in range(100):
                        child_id = identity(child.pid)
                        if child_id and 'medtoolbench.supplement.qwen_kernel_probe run ' in child_id['cmdline']:
                            break
                        time.sleep(.01)
                    atomic_json(control / 'active_probe.json', child_id)
                    while child.poll() is None:
                        if time.time() > window['deadline'] or (control / 'dispatch_resumed.json').exists():
                            stop_probe(control)
                            raise RuntimeError('Bounded diagnostic window expired')
                        atomic_json(control / 'status.json', {'state': 'probing_'+backend, 'time': time.time(),
                            'worker': lane['worker']['name'], 'probe_pid': child.pid,
                            'progress': read(control/backend/'progress.json'), 'formal_dispatch_paused': True})
                        time.sleep(5)
                    if child.returncode or not (read(control/backend/'complete.json') or {}).get('all_pass'):
                        raise RuntimeError('Kernel '+backend+' validation failed; evidence retained')
            comparison = probe.compare(read(control/'reference/comparison_inputs.json'), read(control/'fast/comparison_inputs.json'))
            comparison['stress'] = read(control/'fast/stress.json')
            comparison['time'] = time.time()
            atomic_json(control/'comparison.json', comparison)
            atomic_json(control/'status.json', {'state': 'validated_probe' if comparison['all_outputs_identical'] else 'needs_review',
                'time': time.time(), 'comparison': comparison, 'formal_cutover_authorized': False})
        finally:
            stop_probe(control)
            if paused:
                resume(control, parent)
            elif not (control/'dispatch_resumed.json').exists():
                # Disarm watchdog even if setup failed before SIGSTOP.
                atomic_json(control/'dispatch_resumed.json', {'time': time.time(), 'never_paused': True})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('run', 'watchdog', 'validate'))
    p.add_argument('--control', type=Path, default=probe.CONTROL)
    a = p.parse_args()
    control = a.control.resolve()
    if a.command == 'validate':
        import unittest
        before = probe.source_hashes()
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(
            str(original.ROOT/'repo/tests'), pattern='test*.py'))
        if not result.wasSuccessful() or result.skipped or before != probe.source_hashes():
            raise RuntimeError('Full source-bound regression required')
        atomic_json(control/'unit.json', {'all_pass': True, 'tests': result.testsRun,
            'skipped': result.skipped, 'source_hashes': before, 'time': time.time()})
        return
    if a.command == 'watchdog':
        watchdog(control)
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Diagnostic window interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    with (control/'driver.log').open('x') as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            run(control)
        except BaseException:
            atomic_json(control/'driver_failure.json', {'error': traceback.format_exc(), 'time': time.time()})
            raise


if __name__ == '__main__':
    main()
