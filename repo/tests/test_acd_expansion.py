import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.supplement import acd_expansion as expansion
from medtoolbench.supplement import acd_expansion_runner as entry
from medtoolbench.supplement.storage_recovery import atomic_json, read


class ACDExpansionTests(unittest.TestCase):
    def job(self, name='qualification--qwen35_9b--00', stage='qualification', model='qwen35_9b'):
        return {'id': name, 'model': model, 'stage': stage, 'arm': 'matched_controls',
            'episodes': [{'task': 'task', 'key': 'task--clean--vanilla', 'hazard': 'clean', 'condition': 'vanilla'}],
            'settings': {'seed': 1, 'max_turns': 12, 'max_new_tokens': 2048,
                         'raw_call_budget': 48, 'temperature': 0, 'stress': {}}}

    def plan(self):
        return {'plan_hash': 'parent', 'systems': [{'name': 'qwen35_9b'}, {'name': 'gemma3_12b'}],
            'jobs': [self.job(), self.job('formal--qwen35_9b--00', 'test'),
                     self.job('adverse', 'test', 'llama31_8b')],
            'source_hashes': {}, 'input_hashes': {}, 'require_oracle': True,
            'runner_module': 'medtoolbench.supplement.corrected_runner', 'tasks': [{'id': 'task'}]}

    def manifest(self, cp, new=True):
        return {'new_campaigns': [str(cp)] if new else [], 'campaign_roles': {str(cp): 'A'},
                'repair_jobs': {}, 'allowed_models': ['qwen35_9b', 'ministral3_8b', 'llama31_8b', 'phi4_mini']}

    def test_full_plan_preserves_every_scientific_assignment(self):
        parent = self.plan(); before = copy.deepcopy(parent)
        new = expansion.expanded_plan(parent, Path('/x/supplement_a_v3/plan.json'),
                                      {'name': 'llama31_8b'}, [], {})
        self.assertEqual(parent, before)
        self.assertEqual(len(new['jobs']), 2)
        for original, added in zip(parent['jobs'][:2], new['jobs']):
            self.assertEqual(expansion.scientific_job(original), expansion.scientific_job(added))
            self.assertTrue(added['id'].startswith('full--'))
            self.assertEqual(added['model'], 'llama31_8b')
        self.assertFalse(new['require_oracle'])
        self.assertFalse(new['acd_expansion']['old_adverse_arm_pooled'])
        self.assertTrue(new['acd_expansion']['clinical_accuracy_not_a_gate'])
        self.assertEqual(new['tasks'], parent['tasks'])

    def test_new_model_and_b_expansions_are_rejected(self):
        for model in ('gemma3_12b', 'gptoss20b', 'phi4_mini', 'qwen3_8b'):
            with self.assertRaises(RuntimeError):
                expansion.expanded_plan(self.plan(), Path('/x/supplement_a_v3/plan.json'), {'name': model}, [], {})
        with self.assertRaises(RuntimeError):
            expansion.expanded_plan(self.plan(), Path('/x/supplement_b_v2/plan.json'), {'name': 'llama31_8b'}, [], {})

    def test_expansion_requires_fresh_development_jobs(self):
        p = self.plan(); p['jobs'] = [j for j in p['jobs'] if j['stage'] != 'qualification']
        with self.assertRaisesRegex(RuntimeError, 'development'):
            expansion.expanded_plan(p, Path('/x/supplement_c_v4/plan.json'), {'name': 'llama31_8b'}, [], {})

    def test_original_overlay_preserved(self):
        cp = Path('/original/supplement_a_v3/plan.json'); job = self.job('concurrent')
        m = self.manifest(cp); m['repair_jobs'][str(cp) + '::concurrent'] = {}
        m['repair_control'] = '/original/repair'
        self.assertEqual(expansion.folder_for(Path('/new'), cp, job, m),
                         Path('/original/repair/repair_jobs/concurrent'))

    def test_never_dispatch_b_or_retired_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'campaign_a/plan.json'; bp = Path(tmp) / 'supplement_b_v2/plan.json'
            p = {'jobs': [self.job(model=m) for m in ('llama31_8b', 'gptoss20b', 'gemma3_12b')]}
            m = self.manifest(cp)
            ready = expansion.eligible(Path(tmp), [(cp, p), (bp, p)], m, set(), {})
            self.assertEqual(len(ready), 1)
            self.assertEqual(ready[0][3], cp)
            self.assertEqual(ready[0][5]['model'], 'llama31_8b')

    def test_qualification_precedes_formal_and_uses_same_model_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'plan.json'; j = self.job('formal', 'test', 'llama31_8b'); m = self.manifest(cp)
            with patch.object(expansion, 'model_gate', return_value='waiting'):
                self.assertFalse(expansion.eligible(Path(tmp), [(cp, {'jobs': [j]})], m, set(), {}))
            with patch.object(expansion, 'model_gate', return_value='passed') as gate:
                ready = expansion.eligible(Path(tmp), [(cp, {'jobs': [j]})], m, set(), {})
                self.assertEqual(len(ready), 1)
                self.assertEqual(gate.call_args.args[2], 'llama31_8b')

    def test_existing_acd_formal_work_has_priority_over_added_formal_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / 'supplement_a_v3/plan.json'; new = Path(tmp) / 'campaign_a/plan.json'
            m = self.manifest(new); m['campaign_roles'][str(old)] = 'A'
            plans = [(new, {'jobs': [self.job('new', 'test', 'llama31_8b')]}),
                     (old, {'jobs': [self.job('old', 'test')]})]
            with patch.object(expansion, 'model_gate', return_value='passed'):
                ready = expansion.eligible(Path(tmp), plans, m, set(), {})
            self.assertEqual(sorted(ready, key=lambda r: r[:3])[0][5]['id'], 'old')

    def test_old_adverse_qualification_cannot_qualify_new_full_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'campaign_a/plan.json'
            p = {'require_oracle': False, 'jobs': [self.job('full-dev', model='llama31_8b'),
                                                 self.job('full-test', 'test', 'llama31_8b')]}
            atomic_json(Path(tmp) / 'old/jobs/adverse-dev/result.json', {'state': 'complete'})
            self.assertEqual(expansion.model_gate(cp, p, 'llama31_8b'), 'waiting')

    def test_partial_completed_and_active_jobs_are_not_relaunched(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'campaign_a/plan.json'; m = self.manifest(cp)
            jobs = [self.job(name, model='llama31_8b') for name in ('partial', 'done', 'live')]
            (cp.parent / 'jobs/partial/attempt1').mkdir(parents=True)
            atomic_json(cp.parent / 'jobs/done/result.json', {'state': 'complete', 'episodes': 1})
            self.assertFalse(expansion.eligible(Path(tmp), [(cp, {'jobs': jobs})], m, {(str(cp), 'live')}, {}))

    def test_failed_llama_is_quarantined_without_blocking_other_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'campaign_a/plan.json'; m = self.manifest(cp)
            jobs = [self.job('failed', model='llama31_8b'), self.job('llama-next', model='llama31_8b'), self.job('healthy')]
            atomic_json(cp.parent / 'jobs/failed/result.json', {'state': 'quarantined'})
            plans = [(cp, {'jobs': jobs})]
            pending, blocked, _ = expansion.inventory(Path(tmp), plans, m)
            self.assertEqual(len(pending), 3)
            ready = expansion.eligible(Path(tmp), plans, m, set(), blocked)
            self.assertEqual([r[5]['id'] for r in ready], ['healthy'])

    def test_b_failure_does_not_block_acd(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'campaign_a/plan.json'; bp = Path(tmp) / 'supplement_b_v2/plan.json'
            atomic_json(bp.parent / 'jobs/b/result.json', {'state': 'quarantined'})
            jobs = [(cp, {'jobs': [self.job('a')]}), (bp, {'jobs': [self.job('b')]})]
            pending, blocked, _ = expansion.inventory(Path(tmp), jobs, self.manifest(cp))
            self.assertFalse(blocked)
            self.assertEqual([j['job'] for j in pending], ['a'])

    def test_concurrent_full_job_remains_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / 'campaign_a/plan.json'; j = self.job('full-concurrent', 'test', 'llama31_8b')
            j['settings']['stress'] = {'concurrent': True}
            with patch.object(expansion, 'model_gate', return_value='passed'):
                self.assertEqual(len(expansion.eligible(Path(tmp), [(cp, {'jobs': [j]})],
                                     self.manifest(cp), set(), {})), 1)

    def test_dispatch_scope_guard_prevents_direct_b_attempt(self):
        with patch.object(expansion, 'load', return_value={'campaign_roles': {}}), patch.object(expansion, 'execute') as execute:
            with self.assertRaisesRegex(RuntimeError, 'scope'):
                expansion.attempt(Path('/manifest.json'), Path('/b/plan.json'), self.job(), None, None, None)
            execute.assert_not_called()

    def test_direct_formal_runner_requires_development_gate(self):
        cp = Path('/test/campaign_a/plan.json'); job = self.job('formal', 'test', 'llama31_8b')
        manifest = {'new_campaigns': [str(cp)], 'plan_hashes': {str(cp): 'p'}}
        plan = {'plan_hash': 'p', 'jobs': [job]}
        argv = ['runner', '--expansion-manifest', '/test/manifest.json', '--plan', str(cp), '--job', 'formal']
        with patch.object(sys, 'argv', argv), patch.object(entry, 'load', return_value=manifest), \
             patch.object(entry, 'load_plan', return_value=plan), patch.object(entry, 'model_gate', return_value='waiting'):
            with self.assertRaisesRegex(RuntimeError, 'qualification'):
                entry.main()

    def test_runner_records_scope_and_installs_only_certified_components(self):
        from medtoolbench.supplement import runner, corrected_runner, profile_runner
        from medtoolbench.supplement.concurrent_repair import ConcurrentRejectionProxy
        from medtoolbench.supplement.profile_store import ProfilePreservingProcess
        from medtoolbench.research import run as research_run
        original_parser = research_run.parse_action
        original_model = runner.SampleModel
        for concurrent, profile in ((False, False), (True, False), (False, True)):
            with self.subTest(concurrent=concurrent, profile=profile), tempfile.TemporaryDirectory() as tmp:
                cp = Path(tmp) / 'plan.json'; job = self.job('dev', model='llama31_8b')
                job['settings']['stress'] = {'concurrent': concurrent}
                module = profile_runner if profile else corrected_runner
                plan = {'plan_hash': 'p', 'jobs': [job], 'runner_module': module.__name__}
                manifest = {'new_campaigns': [str(cp)], 'plan_hashes': {str(cp): 'p'},
                            'manifest_hash': 'm', 'source_hashes': {}}
                output = Path(tmp) / 'config.json'
                def run_without_model():
                    self.assertIsNot(runner.SampleModel, original_model)
                    if concurrent: self.assertIs(runner.StressProxy, ConcurrentRejectionProxy)
                    if profile: self.assertIs(runner.HapiProcess, ProfilePreservingProcess)
                    self.assertIsNot(research_run.parse_action, original_parser)
                    runner.write_json(output, {'job': job})
                argv = ['runner', '--expansion-manifest', str(Path(tmp) / 'manifest.json'),
                        '--plan', str(cp), '--job', 'dev']
                with patch.object(sys, 'argv', argv), patch.object(entry, 'load', return_value=manifest), \
                     patch.object(entry, 'load_plan', return_value=plan), patch.object(runner, 'main', side_effect=run_without_model):
                    entry.main()
                record = read(output)['acd_expansion']
                self.assertEqual(record['manifest_hash'], 'm')
                self.assertFalse(record['json_envelope_candidate_enabled'])
                self.assertFalse(record['payload_corrected'])
                self.assertEqual(bool(record['concurrent_repair']), concurrent)
                self.assertIs(runner.SampleModel, original_model)
                self.assertIs(research_run.parse_action, original_parser)


if __name__ == '__main__':
    unittest.main()
