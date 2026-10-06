import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from medtoolbench.storage import digest
from medtoolbench.research.build import save_gzip
from medtoolbench.supplement import gptoss_compile as compiler
from medtoolbench.supplement import gptoss_compile_recovery as recovery
from medtoolbench.supplement import gptoss_compile_runner as entry
from medtoolbench.supplement.storage_recovery import atomic_json, read


class CompilerTests(unittest.TestCase):
    def test_capacity_is_explicit_and_eager_fallback_fails_closed(self):
        config = SimpleNamespace(suppress_errors=True, recompile_limit=8,
            accumulated_recompile_limit=256, fail_on_recompile_limit_hit=False)
        proof = compiler.configure(config)
        self.assertEqual(config.recompile_limit, 256)
        self.assertEqual(config.accumulated_recompile_limit, 4096)
        self.assertTrue(config.fail_on_recompile_limit_hit)
        self.assertFalse(config.suppress_errors)
        self.assertTrue(proof['weights_precision_batch_budgets_prompts_unchanged'])

    def test_model_inherits_all_generation_and_prompt_implementations(self):
        from medtoolbench.supplement.gptoss_native_model import NativeModel
        klass = compiler.repaired_model_class()
        self.assertIs(klass.generate_sample, NativeModel.generate_sample)
        self.assertIs(klass.prompt, NativeModel.prompt)
        def init(self, *args, **kwargs):
            self.metadata = {}; self.args = (args, kwargs)
        with patch.object(compiler, 'install', return_value={'revision': compiler.REVISION}), patch.object(NativeModel, '__init__', init):
            model = klass('synthetic', thinking=True, seed=77)
        self.assertEqual(model.args, (('synthetic',), {'thinking': True, 'seed': 77}))
        self.assertFalse(model.metadata['inherited_compute_unchanged'])
        self.assertEqual(model.metadata['compiler_capacity_repair']['revision'], compiler.REVISION)


class OomRecoveryTests(unittest.TestCase):
    def incident(self, root):
        job = {'id':'native--matched_controls--gptoss20b--00', 'stage':'test', 'model':'gptoss20b',
               'episodes':[{'key':'synthetic'}]}
        folder = root/'old'/'jobs'/job['id']; out = folder/'attempt1'; out.mkdir(parents=True)
        (out/'episodes').mkdir()
        plan = {'plan_hash':'synthetic', 'jobs':[job]}
        row = {'job':job['id'], 'state':'quarantined', 'run':str(out),
               'attempts':[{'returncode':1, 'timeout':None}]}
        atomic_json(folder/'result.json', row)
        atomic_json(out/'config.json', {'plan_hash':plan['plan_hash'], 'job':job})
        atomic_json(out/'failure.json', {'error':'math_attention\ntorch.OutOfMemoryError: CUDA out of memory'})
        atomic_json(out/'progress.json', {'completed':38})
        for i in range(38):
            save_gzip(out/'episodes'/('%02d.json.gz' % i), {'synthetic':i})
        (folder/'attempt1.replay.log').write_text('torch._dynamo hit config.recompile_limit (8)')
        return folder,out,plan,job,row

    def test_only_known_oom_scope_is_authorized_and_all_partials_archived(self):
        with tempfile.TemporaryDirectory() as d:
            folder,out,plan,job,row = self.incident(Path(d))
            before = recovery.evidence_hashes(out)
            proof = recovery.validate_oom(folder,plan,job)
            self.assertEqual(proof['partial_episodes_excluded'],38)
            self.assertFalse(proof['prior_partial_rows_pooled'])
            self.assertEqual(before,recovery.evidence_hashes(out))
            self.assertEqual(read(folder/'result.json'),row)

    def test_non_oom_audit_failure_and_completed_runs_never_retry(self):
        with tempfile.TemporaryDirectory() as d:
            folder,out,plan,job,row = self.incident(Path(d))
            atomic_json(out/'failure.json',{'error':'Native input replay mismatch'})
            with self.assertRaisesRegex(RuntimeError,'compiler-fallback'):
                recovery.validate_oom(folder,plan,job)
            row['state']='complete'; atomic_json(folder/'result.json',row)
            with self.assertRaisesRegex(RuntimeError,'Only the two'):
                recovery.validate_oom(folder,plan,job)

    def test_any_existing_audit_or_completion_blocks_retry(self):
        for filename in ('audit.json','complete.json','native_replay_audit.json'):
            with tempfile.TemporaryDirectory() as d:
                folder,out,plan,job,row = self.incident(Path(d))
                atomic_json(out/filename,{'all_pass':False})
                with self.assertRaisesRegex(RuntimeError,'Audited/complete'):
                    recovery.validate_oom(folder,plan,job)

    def test_missing_compiler_cause_or_partial_count_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder,out,plan,job,row = self.incident(Path(d))
            atomic_json(out/'progress.json',{'completed':37})
            with self.assertRaisesRegex(RuntimeError,'partial evidence'):
                recovery.validate_oom(folder,plan,job)
            atomic_json(out/'progress.json',{'completed':38})
            (folder/'attempt1.replay.log').write_text('some unrelated failure')
            with self.assertRaisesRegex(RuntimeError,'compiler-fallback'):
                recovery.validate_oom(folder,plan,job)

    def test_identity_timeout_extra_attempt_and_wrong_model_rejected(self):
        for kind in ('identity','timeout','extra_attempt','model'):
            with tempfile.TemporaryDirectory() as d:
                folder,out,plan,job,row = self.incident(Path(d))
                if kind=='identity':
                    atomic_json(out/'config.json',{'plan_hash':'changed','job':job})
                elif kind=='timeout':
                    row['attempts'][0]['timeout']='wall time'; atomic_json(folder/'result.json',row)
                elif kind=='extra_attempt':
                    (folder/'attempt2').mkdir()
                else:
                    job['model']='another-model'
                with self.assertRaises(RuntimeError):
                    recovery.validate_oom(folder,plan,job)

    def scope(self, root):
        cp = root/'new/campaign_a/plan.json'; old = root/'old/jobs/job/result.json'
        job={'id':'job','stage':'test','model':'gptoss20b'}
        plan={'plan_hash':'same','jobs':[job]}
        m={'campaigns':[str(cp)],'plan_hashes':{str(cp):'same'},
           'compiler_history':[{'campaign':str(cp),'job':'job','policy':'unstarted','original_result':str(old)}]}
        return cp,job,plan,m

    def test_unstarted_formal_authorized_only_after_development_gate(self):
        with tempfile.TemporaryDirectory() as d:
            cp,job,plan,m=self.scope(Path(d))
            with patch.object(recovery.replay,'development_passed',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'development'):
                    recovery.authorize_job(m,cp,plan,job)
            with patch.object(recovery.replay,'development_passed',return_value=True):
                recovery.authorize_job(m,cp,plan,job)

    def test_completed_or_started_inference_cannot_be_rerun(self):
        for kind in ('adopted','result','attempt','log','old_attempt'):
            with tempfile.TemporaryDirectory() as d:
                cp,job,plan,m=self.scope(Path(d)); folder=cp.parent/'jobs'/job['id']
                folder.mkdir(parents=True)
                if kind=='adopted': m['compiler_history'][0]['policy']='reuse_complete'
                elif kind=='result': atomic_json(folder/'result.json',{'state':'complete'})
                elif kind=='attempt': (folder/'attempt1').mkdir()
                elif kind=='log': (folder/'attempt1.compiler.log').touch()
                else:
                    old=Path(m['compiler_history'][0]['original_result']).parent
                    (old/'attempt1').mkdir(parents=True)
                with self.assertRaises(RuntimeError):
                    recovery.authorize_job(m,cp,plan,job)

    def test_child_may_accept_only_own_open_launch_log(self):
        with tempfile.TemporaryDirectory() as d:
            cp,job,plan,m=self.scope(Path(d)); folder=cp.parent/'jobs'/job['id']; folder.mkdir(parents=True)
            (folder/'attempt1.compiler.log').touch()
            with patch.object(recovery.replay,'development_passed',return_value=True):
                recovery.authorize_job(m,cp,plan,job,allow_launch_log=True)
                (folder/'attempt1').mkdir()
                with self.assertRaisesRegex(RuntimeError,'already started'):
                    recovery.authorize_job(m,cp,plan,job,allow_launch_log=True)

    def test_retry_authorization_requires_unchanged_failure_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            cp,job,plan,m=self.scope(Path(d)); row=m['compiler_history'][0]
            row.update(policy='one_oom_retry',authorization={'sentinel':1})
            with patch.object(recovery,'validate_oom',return_value={'sentinel':2}):
                with self.assertRaisesRegex(RuntimeError,'OOM evidence changed'):
                    recovery.authorize_job(m,cp,plan,job)

    def test_gpu_or_cpu_proofs_cannot_be_missing(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'regression'):
                recovery.validate_proofs(Path(d)/'unit.json',Path(d)/'gpu.json')

    def test_runner_keeps_native_parser_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as d:
            cp,job,plan,m=self.scope(Path(d)); out=cp.parent/'jobs'/job['id']/'attempt1'
            argv=['runner','--compiler-manifest',str(Path(d)/'manifest.json'),'--plan',str(cp),
                  '--job','job','--worker','synthetic-gpu','--out',str(out)]
            sentinel=type('SyntheticModel',(),{})
            with patch.object(sys,'argv',argv), patch.object(entry,'load',return_value=m), \
                 patch.object(entry,'load_plan',return_value=plan), patch.object(entry,'authorize_job') as auth, \
                 patch.object(entry,'repaired_model_class',return_value=sentinel), \
                 patch.object(entry.native,'load'), patch.object(entry.native,'development_passed'), \
                 patch.object(entry.entry,'NativeModel'), patch.object(entry.entry,'audit_native'), \
                 patch.object(entry.entry,'main') as launch:
                entry.main()
                auth.assert_called_once_with(m,cp,plan,job,allow_launch_log=True)
                self.assertIs(entry.entry.NativeModel,sentinel)
                self.assertIs(entry.entry.audit_native,entry.audit_native)
                launch.assert_called_once()


if __name__=='__main__':
    unittest.main()
