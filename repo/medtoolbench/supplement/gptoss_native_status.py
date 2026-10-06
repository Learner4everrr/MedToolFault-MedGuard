"""Read-only, fresh native cohort status; old GPT-OSS rows are never counted."""
import argparse
import json
import time
from pathlib import Path

from .storage_recovery import read


def snapshot(control=None):
    if control is None:
        compiler = Path('data/gptoss_compile_recovery_v1')
        recovery = Path('data/gptoss_native_replay_recovery_v1')
        control = (compiler if (compiler / 'manifest.json').exists() else
                   recovery if (recovery / 'manifest.json').exists() else Path('data/gptoss_acd_native_v3'))
    control = Path(control)
    manifest = read(control / 'manifest.json')
    state = read(control / 'status.json')
    result = {'control': str(control), 'state': state, 'manifest_hash': (manifest or {}).get('manifest_hash'),
        'automatic_b_resume': False, 'gemma_retired': True, 'old_gptoss_pooled': False,
        'complete': read(control / 'complete.json'), 'needs_review': read(control / 'needs_review.json'),
        'bootstrap_failure': read(control / 'bootstrap_failure.json'), 'time': time.time()}
    if manifest and manifest.get('replay_recovery_revision'):
        result.update(audit_recovery_revision=manifest['replay_recovery_revision'],
                      recovered_without_rerunning_inference=manifest['recovered_episodes'],
                      original_failed_receipts_preserved=True)
    if manifest and manifest.get('compiler_recovery_revision'):
        result.update(compiler_recovery_revision=manifest['compiler_recovery_revision'],
                      reused_formal_episodes=manifest['reused_formal_episodes'],
                      archived_partial_episodes_excluded=manifest['archived_partial_episodes'],
                      scientific_protocol_changed=manifest['clinical_protocol_changed'])
    if state and 'heartbeat' in state:
        result['heartbeat_age_seconds'] = time.time() - state['heartbeat']
    counts = {s: {'expected': 0, 'audited': 0, 'saved_partial': 0} for s in ('qualification', 'test')}
    campaigns = {}
    for path in (manifest or {}).get('campaigns', []):
        cp = Path(path); plan = read(cp)
        cc = {s: {'expected': 0, 'audited': 0, 'saved_partial': 0} for s in counts}
        for job in plan['jobs']:
            c = cc[job['stage']]; c['expected'] += len(job['episodes'])
            folder = cp.parent / 'jobs' / job['id']; row = read(folder / 'result.json')
            if row and row.get('state') == 'complete':
                c['audited'] += row['episodes']
            elif not row and not (folder / 'attempt1/failure.json').exists():
                c['saved_partial'] += (read(folder / 'attempt1/progress.json') or {}).get('completed', 0)
        campaigns[manifest['campaign_roles'][path]] = cc
        for stage in counts:
            for key in counts[stage]:
                counts[stage][key] += cc[stage][key]
    result.update(counts=counts, campaigns=campaigns, workers={})
    for name in (state or {}).get('busy', {}):
        row = read(control / 'workers' / (name + '.json'))
        if row:
            result['workers'][name] = {**row, 'progress': read(Path(row['out']) / 'progress.json')}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--control', type=Path)
    print(json.dumps(snapshot(p.parse_args().control), indent=2))
