import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.storage import digest
from medtoolbench.research.build import save_gzip, load_gzip
from medtoolbench.supplement.gptoss_model import FROZEN_DATE
from medtoolbench.supplement.gptoss_native import decode, render
from medtoolbench.supplement.gptoss_replay_audit import audit_row, audit_native, evidence_hashes, REVISION
from medtoolbench.supplement import gptoss_replay_recovery as recovery
from medtoolbench.supplement import gptoss_replay_runner as entry
from medtoolbench.supplement.storage_recovery import atomic_json, read


def generation(raw):
    text, status, metadata = decode(raw)
    return {'raw_harmony': raw, 'text': text, 'harmony_status': status, 'native_transport': metadata}


class ReplayAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (recovery.ROOT / 'data/models/gpt-oss-20b/tokenizer_config.json').is_file():
            raise unittest.SkipTest('Optional pinned GPT-OSS tokenizer is not bundled in the code release')
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(str(recovery.ROOT / 'data/models/gpt-oss-20b'), local_files_only=True)
        cls.template = cls.tok.chat_template.replace('strftime_now("%Y-%m-%d")', json.dumps(FROZEN_DATE))

    def row(self):
        action = {'method': 'POST', 'path': '/Observation', 'body': {
            'resourceType': 'Observation', 'status': 'final', 'valueQuantity': {'value': 8, 'unit': 'test'},
            'component': [{'z': 1, 'a': 2}], 'code': {'text': 'synthetic'}}}
        call = generation(' to=functions.fhir_request<|channel|>commentary json<|message|>' + json.dumps(action) + '<|call|>')
        final = generation('<|channel|>final<|message|>{"final":{"id":"synthetic"}}<|return|>')
        messages = [{'role': 'system', 'content': 'Original synthetic instructions.'},
                    {'role': 'user', 'content': 'Synthetic post and feedback.'}]
        for g in (call, final):
            tokens = render(self.tok, self.template, messages)
            g.update(input_tokens=len(tokens), native_prompt_hash=digest(tokens))
            messages.append({'role': 'assistant', 'content': g['text'], 'native_transport': copy.deepcopy(g['native_transport'])})
            if g is call:
                messages.append({'role': 'user', 'content': '{"status":201,"body":{"resourceType":"Observation","id":"synthetic"}}'})
        return {'key': 'synthetic--clean--vanilla', 'messages': messages, 'generations': [call, final]}

    def plan(self):
        return {'plan_hash': 'synthetic-only', 'systems': [{'path': str(recovery.ROOT / 'data/models/gpt-oss-20b')}],
                'jobs': [{'id': 'synthetic', 'episodes': [{'key': 'synthetic--clean--vanilla'}]}]}

    def test_actual_gzip_roundtrip_nested_key_order_recovers_exact_hashes(self):
        from medtoolbench.supplement.gptoss_native_runner import audit_native as old_audit
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); (out / 'episodes').mkdir()
            row, plan = self.row(), self.plan()
            atomic_json(out / 'config.json', {'plan_hash': plan['plan_hash'], 'job': plan['jobs'][0]})
            save_gzip(out / 'episodes/synthetic.json.gz', row)
            loaded = load_gzip(out / 'episodes/synthetic.json.gz')
            self.assertNotEqual(list(row['messages'][2]['native_transport']['arguments']),
                                list(loaded['messages'][2]['native_transport']['arguments']))
            with self.assertRaisesRegex(RuntimeError, 'input/history replay mismatch'):
                old_audit(out, plan)
            before = evidence_hashes(out)
            certificate = audit_native(out, plan, tokenizer=(self.tok, self.template))
            self.assertTrue(certificate['all_pass'])
            self.assertEqual(certificate['generations'], 2)
            self.assertEqual(before, evidence_hashes(out))
            self.assertFalse(certificate['inference_changed'])

    def test_raw_evidence_is_not_mutated_even_in_memory(self):
        row = json.loads(json.dumps(self.row(), sort_keys=True)); before = copy.deepcopy(row)
        self.assertEqual(sum(audit_row(row, self.tok, self.template).values()), 2)
        self.assertEqual(row, before)

    def test_changed_prompt_hash_or_count_fails_not_length_only(self):
        for field, value in (('native_prompt_hash', 'bad'), ('input_tokens', 1)):
            row = self.row(); row['generations'][1][field] = value
            with self.assertRaisesRegex(RuntimeError, 'Exact native input replay mismatch'):
                audit_row(row, self.tok, self.template)

    def test_changed_response_is_not_reconstructed_or_repaired(self):
        row = self.row(); row['messages'][3]['content'] = '{"status":400,"body":{}}'
        with self.assertRaisesRegex(RuntimeError, 'Exact native input replay mismatch'):
            audit_row(row, self.tok, self.template)

    def test_changed_arguments_or_raw_text_fail_closed(self):
        for kind in ('arguments', 'raw', 'message'):
            row = self.row()
            if kind == 'arguments':
                row['generations'][0]['native_transport']['arguments']['method'] = 'DELETE'
            elif kind == 'raw':
                row['generations'][0]['raw_harmony'] = row['generations'][0]['raw_harmony'].replace('POST', 'DELETE')
            else:
                row['messages'][2]['native_transport']['arguments']['method'] = 'DELETE'
            with self.assertRaises(RuntimeError):
                audit_row(row, self.tok, self.template)

    def test_unexpected_tail_not_ignored(self):
        row = self.row(); row['messages'].append({'role': 'assistant', 'content': 'extra'})
        with self.assertRaisesRegex(RuntimeError, 'conversation tail'):
            audit_row(row, self.tok, self.template)

    def test_coverage_and_identity_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); (out / 'episodes').mkdir(); plan = self.plan()
            atomic_json(out / 'config.json', {'plan_hash': plan['plan_hash'], 'job': plan['jobs'][0]})
            with self.assertRaisesRegex(RuntimeError, 'coverage'):
                audit_native(out, plan, tokenizer=(self.tok, self.template))
            save_gzip(out / 'episodes/one.json.gz', self.row())
            save_gzip(out / 'episodes/two.json.gz', self.row())
            with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
                audit_native(out, plan, tokenizer=(self.tok, self.template))
            atomic_json(out / 'config.json', {'plan_hash': 'other', 'job': plan['jobs'][0]})
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                audit_native(out, plan, tokenizer=(self.tok, self.template))

    def test_existing_certificate_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); atomic_json(out / 'native_replay_audit.json', {'sentinel': True})
            with self.assertRaisesRegex(RuntimeError, 'overwrite'):
                audit_native(out, self.plan(), tokenizer=(self.tok, self.template))
            self.assertEqual(read(out / 'native_replay_audit.json'), {'sentinel': True})


class RecoveryTests(unittest.TestCase):
    def job(self, stage='qualification'):
        return {'id': 'job', 'model': 'gptoss20b', 'stage': stage, 'episodes': [{'key': 'synthetic'}]}

    def setup_scope(self, root, stage='qualification'):
        cp = root / 'new/campaign_a/plan.json'; old = root / 'old/campaign_a/plan.json'
        job = self.job(stage); plan = {'plan_hash': 'unchanged', 'jobs': [job]}
        manifest = {'campaigns': [str(cp)], 'original_campaigns': {str(cp): str(old)},
                    'plan_hashes': {str(cp): 'unchanged'}}
        return cp, old, job, plan, manifest

    def test_only_exact_audit_failure_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'job'; out = folder / 'attempt1'; out.mkdir(parents=True)
            result = {'run': str(out), 'state': 'quarantined', 'attempts': [{'returncode': 1, 'timeout': None}]}
            atomic_json(folder / 'result.json', result)
            log = folder / 'attempt1.native.log'
            log.write_text('RuntimeError: Native input/history replay mismatch: synthetic')
            with patch.object(recovery, 'verify_completion'):
                self.assertEqual(recovery.recovery_eligible(folder, {}, self.job()), result)
                log.write_text('CUDA out of memory')
                with self.assertRaisesRegex(RuntimeError, 'Unreviewed'):
                    recovery.recovery_eligible(folder, {}, self.job())

    def test_formal_or_other_model_failure_not_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); (folder / 'attempt1').mkdir()
            atomic_json(folder / 'result.json', {'state': 'quarantined'})
            for job in (self.job('test'), dict(self.job(), model='gemma3_12b')):
                with self.assertRaisesRegex(RuntimeError, 'Only the completed'):
                    recovery.recovery_eligible(folder, {}, job)

    def test_partial_or_clinical_audit_failure_not_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); out = folder / 'attempt1'; out.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'Partial'):
                recovery.recovery_eligible(folder, {}, self.job())
            atomic_json(folder / 'result.json', {'state': 'quarantined', 'run': str(out)})
            with patch.object(recovery, 'verify_completion', side_effect=RuntimeError('clinical audit invalid')):
                with self.assertRaisesRegex(RuntimeError, 'clinical audit invalid'):
                    recovery.recovery_eligible(folder, {}, self.job())

    def test_new_development_allowed_formal_requires_global_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            cp, old, job, plan, manifest = self.setup_scope(Path(directory))
            recovery.authorize_job(manifest, cp, plan, job)
            job['stage'] = 'test'
            with patch.object(recovery, 'development_passed', return_value=False):
                with self.assertRaisesRegex(RuntimeError, '260'):
                    recovery.authorize_job(manifest, cp, plan, job)
            with patch.object(recovery, 'development_passed', return_value=True):
                recovery.authorize_job(manifest, cp, plan, job)

    def test_old_or_new_started_inference_never_relaunched(self):
        for where in ('old', 'new'):
            with tempfile.TemporaryDirectory() as directory:
                cp, old, job, plan, manifest = self.setup_scope(Path(directory))
                target = old if where == 'old' else cp
                (target.parent / 'jobs/job/attempt1').mkdir(parents=True)
                with self.assertRaisesRegex(RuntimeError, 'cannot run again'):
                    recovery.authorize_job(manifest, cp, plan, job)

    def test_b_or_other_model_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cp, old, job, plan, manifest = self.setup_scope(Path(directory))
            with self.assertRaisesRegex(RuntimeError, 'scope'):
                recovery.authorize_job(manifest, cp.parent / 'b.json', plan, job)
            job['model'] = 'gemma3_12b'
            with self.assertRaisesRegex(RuntimeError, 'scope'):
                recovery.authorize_job(manifest, cp, plan, job)

    def test_only_child_can_accept_its_already_opened_launch_log(self):
        with tempfile.TemporaryDirectory() as directory:
            cp, old, job, plan, manifest = self.setup_scope(Path(directory))
            folder = cp.parent / 'jobs/job'; folder.mkdir(parents=True)
            (folder / 'attempt1.replay.log').touch()
            with self.assertRaisesRegex(RuntimeError, 'cannot run again'):
                recovery.authorize_job(manifest, cp, plan, job)
            recovery.authorize_job(manifest, cp, plan, job, allow_launch_log=True)
            (folder / 'attempt1').mkdir()
            with self.assertRaisesRegex(RuntimeError, 'cannot run again'):
                recovery.authorize_job(manifest, cp, plan, job, allow_launch_log=True)

    def test_global_gate_does_not_count_original_quarantined_results(self):
        with tempfile.TemporaryDirectory() as directory:
            cp, old, job, plan, manifest = self.setup_scope(Path(directory))
            atomic_json(old.parent / 'jobs/job/result.json', {'state': 'complete', 'episodes': 260})
            with patch.object(recovery, 'load_plan', return_value=plan):
                self.assertFalse(recovery.development_passed(manifest))
            atomic_json(cp.parent / 'jobs/job/result.json', {'state': 'quarantined', 'episodes': 260})
            with patch.object(recovery, 'load_plan', return_value=plan):
                self.assertFalse(recovery.development_passed(manifest))

    def test_certificate_revision_and_hash_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cp, old, job, plan, manifest = self.setup_scope(root)
            out = root / 'run'; cert = root / 'cert.json'
            proof = {'all_pass': True, 'revision': REVISION, 'plan_hash': plan['plan_hash'],
                     'job': 'job', 'episodes': 1, 'run': str(out.resolve())}
            atomic_json(cert, proof)
            result = {'state': 'complete', 'run': str(out), 'replay_audit_path': str(cert), 'replay_audit_hash': digest(proof)}
            with patch.object(recovery, 'verify_completion'):
                self.assertEqual(recovery.verify_result(result, plan, job), proof)
                result['replay_audit_hash'] = 'tampered'
                with self.assertRaisesRegex(RuntimeError, 'Corrected native audit'):
                    recovery.verify_result(result, plan, job)

    def test_runner_replaces_audit_and_gate_only(self):
        with tempfile.TemporaryDirectory() as directory:
            cp, old, job, plan, manifest = self.setup_scope(Path(directory))
            argv = ['runner', '--replay-manifest', str(Path(directory) / 'manifest.json'),
                    '--plan', str(cp), '--job', 'job', '--worker', 'worker', '--out', str(cp.parent / 'jobs/job/attempt1')]
            original_load, original_gate, original_audit = entry.native.load, entry.native.development_passed, entry.entry.audit_native
            try:
                with patch.object(sys, 'argv', argv), patch.object(entry, 'load', return_value=manifest), \
                     patch.object(entry, 'load_plan', return_value=plan), patch.object(entry.entry, 'main') as main:
                    entry.main()
                    main.assert_called_once()
                    self.assertIs(entry.entry.audit_native, entry.audit_native)
                    self.assertIs(entry.native.development_passed, entry.development_passed)
                    self.assertEqual(sys.argv[1:3], ['--native-manifest', str(Path(directory) / 'manifest.json')])
            finally:
                entry.native.load, entry.native.development_passed, entry.entry.audit_native = original_load, original_gate, original_audit


if __name__ == '__main__':
    unittest.main()
