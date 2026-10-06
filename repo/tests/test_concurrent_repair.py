import copy
import tempfile
import unittest
from pathlib import Path

from medtoolbench.storage import Store, outcome
from medtoolbench.research.contracts import service, state_payload
from medtoolbench.supplement.concurrent_repair import ConcurrentRejectionProxy
from medtoolbench.supplement.corrected_runner import InterfaceConsistentProxy
from medtoolbench.supplement.transport import CONDITIONS, CallBudget, FactorClient


class RejectingStore:
    def __init__(self, store, status=400):
        self.store, self.status, self.calls = store, status, []

    def __getattr__(self, name):
        return getattr(self.store, name)

    def call(self, method, path, body=None):
        self.calls.append((method, path, copy.deepcopy(body)))
        if isinstance(body, dict) and body.get('authoredOn') == 'malformed':
            return outcome(self.status, 'invalid authoredOn')
        return self.store.call(method, path, body)


class ConcurrentRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / 'state.db', [{'resourceType': 'Patient', 'id': 'p'}])
        self.store.types = ('Patient', 'ServiceRequest')
        self.store.base = 'http://127.0.0.1/fhir'
        self.body = service('p', '4548-4')
        self.task = {'patient': 'p', 'hazards': ['clean', 'commit_timeout', 'no_commit_timeout'],
                     'contract': {'required_creations': [self.body]}}

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_old_crash_reproduced_and_repaired_without_payload_changes(self):
        bad = {**self.body, 'authoredOn': 'malformed'}
        store = RejectingStore(self.store)
        with self.assertRaisesRegex(RuntimeError, 'Controlled concurrent write failed'):
            InterfaceConsistentProxy(store, self.task, 'commit_timeout', {'concurrent': True}).call('POST', '/ServiceRequest', bad)
        proxy = ConcurrentRejectionProxy(store, self.task, 'commit_timeout', {'concurrent': True})
        budget = CallBudget(proxy.call, 48)
        before = self.store.snapshot()
        self.assertEqual(budget('POST', '/ServiceRequest', bad)['status'], 400)
        self.assertEqual(store.calls[-1][2], bad)
        self.assertEqual(proxy.trace[-1]['body'], bad)
        self.assertEqual(before, self.store.snapshot())
        self.assertEqual((budget.used, proxy.triggered, proxy.counts['commit_timeout']), (1, 0, 0))
        self.assertFalse(proxy.concurrent_done)
        self.assertFalse(proxy.mutation_states)
        self.assertFalse(proxy.external_ids)
        self.assertEqual(budget('POST', '/ServiceRequest', self.body)['status'], 504)
        self.assertEqual((proxy.triggered, len(proxy.external_ids)), (1, 1))
        self.assertEqual(len(proxy.mutation_states), 1)
        self.assertEqual(proxy.mutation_states[0]['trace_index'], 1)

    def test_server_error_is_not_converted_to_agent_rejection(self):
        for status in (500, 503, 504):
            proxy = ConcurrentRejectionProxy(RejectingStore(self.store, status), self.task,
                                            'commit_timeout', {'concurrent': True})
            with self.assertRaisesRegex(RuntimeError, str(status)):
                proxy.call('POST', '/ServiceRequest', {**self.body, 'authoredOn': 'malformed'})
            self.assertFalse(proxy.trace)

    def test_repeated_rejection_consumes_budget_without_consuming_fault(self):
        proxy = ConcurrentRejectionProxy(RejectingStore(self.store), self.task, 'no_commit_timeout', {'concurrent': True})
        budget = CallBudget(proxy.call, 2)
        bad = {**self.body, 'authoredOn': 'malformed'}
        for _ in range(2):
            self.assertEqual(budget('POST', '/ServiceRequest', bad)['status'], 400)
        self.assertEqual(budget('POST', '/ServiceRequest', self.body)['status'], 429)
        self.assertEqual((budget.used, budget.denied, len(proxy.trace), proxy.triggered), (2, 1, 2, 0))

    def test_all_policies_reject_and_valid_path_matches_frozen(self):
        for policy in CONDITIONS:
            for hazard in ('commit_timeout', 'no_commit_timeout'):
                with self.subTest(policy=policy, hazard=hazard):
                    outcomes = []
                    for cls in (InterfaceConsistentProxy, ConcurrentRejectionProxy):
                        self.store.reset()
                        proxy = cls(RejectingStore(self.store), self.task, hazard, {'concurrent': True})
                        budget = CallBudget(proxy.call, 48)
                        client = FactorClient(budget, policy)
                        if cls is ConcurrentRejectionProxy:
                            self.assertEqual(client.call('POST', '/ServiceRequest', {**self.body, 'authoredOn': 'malformed'})['status'], 400)
                            self.assertFalse(proxy.concurrent_done)
                        result = client.call('POST', '/ServiceRequest', self.body)
                        outcomes.append((result['status'], [state_payload(r) for r in self.store.snapshot()],
                                         proxy.triggered, proxy.counts, len(proxy.external_ids)))
                    self.assertEqual(*outcomes)

    def test_nonconcurrent_path_does_not_use_new_rejection_branch(self):
        for hazard in ('clean', 'commit_timeout'):
            proxy = ConcurrentRejectionProxy(RejectingStore(self.store), self.task, hazard)
            self.assertEqual(proxy.call('POST', '/ServiceRequest', {**self.body, 'authoredOn': 'malformed'})['status'], 400)
            self.assertNotIn('concurrent_write_rejected', proxy.trace[-1])

    def test_unexpected_store_exception_is_not_swallowed(self):
        class Broken(RejectingStore):
            def call(self, *args, **kwargs):
                raise ValueError('unexpected backend')
        proxy = ConcurrentRejectionProxy(Broken(self.store), self.task, 'commit_timeout', {'concurrent': True})
        with self.assertRaisesRegex(ValueError, 'unexpected backend'):
            proxy.call('POST', '/ServiceRequest', self.body)


if __name__ == '__main__':
    unittest.main()
