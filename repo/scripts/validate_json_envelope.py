"""CPU regression plus development-only, non-executing parser replay.

Compare each already generated string, not benchmark answers or success rates.
An accepted replay does not simulate later turns or prove end-to-end speedup.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import unittest

from medtoolbench.storage import digest
from medtoolbench.research.build import load_gzip
from medtoolbench.supplement import json_envelope
from medtoolbench.supplement.external_runner import parse as old_external
from medtoolbench.supplement.input_guard import parse_action as old_fhir
from medtoolbench.supplement.memory_execution import verify_completion
from medtoolbench.supplement.non_gptoss_priority import ALLOWED
from medtoolbench.supplement.runner import load_plan
from medtoolbench.supplement.storage_recovery import atomic_json, read

ROOT = Path(__file__).resolve().parents[2]


def sources():
    files = [Path(__file__).resolve(), Path(json_envelope.__file__).resolve(), ROOT / 'repo/tests/test_json_envelope.py']
    return {str(p): digest(p.read_text()) for p in files}


def result(parser, raw):
    try:
        return True, parser(raw), None
    except (ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
        label = str(error).split(': line ')[0]
        return False, None, label[:180]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    if out.exists():
        raise RuntimeError('Never overwrite validation/replay evidence')
    out.mkdir(parents=True)
    hashes = sources()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'repo/tests'))
    with (out / 'unit.log').open('x') as stream:
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not tests.wasSuccessful() or tests.skipped or sources() != hashes:
        raise RuntimeError('Exact-source full CPU regression required')
    atomic_json(out / 'unit.json', {'all_pass': True, 'tests': tests.testsRun,
        'source_hashes': hashes, 'time': time.time()})
    print(json.dumps({'all_pass': True, 'unit_tests': tests.testsRun}), flush=True)
    rows, changes, inputs = [], [], {}
    for name in ('supplement_a_v3', 'supplement_b_v2', 'supplement_c_v4', 'supplement_d_v3'):
        cp = ROOT / 'data' / name / 'plan.json'
        plan = load_plan(cp)
        inputs[str(cp)] = plan['plan_hash']
        external = plan['runner_module'].endswith('external_runner')
        old, new = (old_external, json_envelope.parse_external) if external else (old_fhir, json_envelope.parse_fhir)
        for model in sorted(ALLOWED):
            counts, errors = Counter(), Counter()
            seconds = Counter()
            episode_count = 0
            for job in plan['jobs']:
                if job['model'] != model or job['stage'] != 'qualification':
                    continue
                file = cp.parent / 'jobs' / job['id'] / 'result.json'
                canonical = read(file)
                if not canonical or canonical.get('state') != 'complete':
                    raise RuntimeError('Only fully audited development jobs can be replayed')
                run = Path(canonical['run'])
                verify_completion(run, plan, job)
                inputs[str(file)] = digest(canonical)
                wanted = {e['key'] for e in job['episodes']}
                for path in sorted((run / 'episodes').glob('*.json.gz')):
                    row = load_gzip(path)
                    if row['key'] not in wanted or row['split'] != 'dev':
                        raise RuntimeError('Non-development or unassigned row in replay')
                    episode_count += 1
                    for index, generation in enumerate(row['generations']):
                        raw = generation['text']
                        before, old_value, old_error = result(old, raw)
                        after, new_value, new_error = result(new, raw)
                        if before and after:
                            state = 'accepted_unchanged' if old_value == new_value else 'accepted_value_changed'
                        elif before:
                            state = 'previously_accepted_now_rejected'
                        elif after:
                            state = 'recovered'
                        else:
                            state = 'still_rejected'
                        counts[state] += 1
                        seconds[state] += generation['batch_seconds_per_item']
                        if not before:
                            errors[old_error] += 1
                        if state in ('recovered', 'previously_accepted_now_rejected', 'accepted_value_changed'):
                            entry = {'source': str(path), 'generation_index': index,
                                'raw_hash': digest(raw), 'state': state, 'old_error': old_error, 'new_error': new_error}
                            if after:
                                envelope = json_envelope.extract_object(raw)
                                entry.update(start=envelope.start, end=envelope.end,
                                    payload_hash=digest(envelope.text), reasoning_prefix_removed=envelope.reasoning_prefix_removed)
                            changes.append(entry)
            if counts:
                old_rejected = counts['recovered'] + counts['still_rejected']
                rows.append({'campaign': name, 'model': model, 'development_episodes': episode_count,
                    'generations': sum(counts.values()), 'counts': dict(counts),
                    'old_rejected_generations': old_rejected,
                    'recovered_fraction_of_old_rejections': counts['recovered'] / old_rejected if old_rejected else None,
                    'amortized_seconds_by_replay_state': dict(seconds),
                    'old_error_types': errors.most_common(), 'actual_speedup_measured': False})
    if sources() != hashes:
        raise RuntimeError('Parser sources changed during replay')
    regression_free = all(not r['counts'].get('accepted_value_changed', 0)
                          and not r['counts'].get('previously_accepted_now_rejected', 0) for r in rows)
    report = {'time': time.time(), 'revision': json_envelope.REVISION, 'source_hashes': hashes,
        'inputs': inputs, 'unit_tests': tests.testsRun, 'regression_free_on_development': regression_free,
        'development_only': True, 'held_out_outputs_used': False, 'reference_answers_used': False,
        'model_inference': False, 'tool_execution': False, 'formal_dispatch_enabled': False,
        'full_trajectory_or_speedup_simulated': False, 'rows': rows, 'changed_parse_records': changes}
    atomic_json(out / 'replay.json', report)
    print(json.dumps({'regression_free_on_development': regression_free,
        'development_episodes': sum(r['development_episodes'] for r in rows),
        'generations': sum(r['generations'] for r in rows),
        'recovered': sum(r['counts'].get('recovered', 0) for r in rows),
        'formal_dispatch_enabled': False}), flush=True)


if __name__ == '__main__':
    main()
