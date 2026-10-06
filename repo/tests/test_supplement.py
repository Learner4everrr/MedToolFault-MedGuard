import copy
import tempfile
import unittest
from pathlib import Path

from medtoolbench.storage import Store, bundle, outcome
from medtoolbench.research.contracts import service
from medtoolbench.supplement.transport import CallBudget, FactorClient, StressProxy
from medtoolbench.supplement.corrected_runner import InterfaceConsistentProxy


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patient = {"resourceType": "Patient", "id": "p"}
        self.lab = {"resourceType": "Observation", "id": "a", "subject": {"reference": "Patient/p"},
            "code": {"coding": [{"code": "K"}]}, "status": "final", "effectiveDateTime": "2023-11-13T09:00:00Z",
            "valueQuantity": {"value": 3, "unit": "mmol/L"}}
        self.store = Store(Path(self.temp.name) / "state.db", [self.patient, self.lab])
        self.store.types = ("Patient", "Observation", "ServiceRequest", "MedicationRequest")
        self.store.base = "http://127.0.0.1/fhir"
        self.task = {"patient": "p", "read_code": "K", "contract": {"required_creations": [service("p", "4548-4")]},
            "hazards": ["clean", "transient_503", "commit_timeout", "no_commit_timeout", "stale", "patient_mismatch", "provenance_conflict"],
            "stale_resource": dict(self.lab, id="old", valueQuantity={"value": 4, "unit": "mmol/L"}),
            "distractor_resources": [dict(self.lab, id="wrong", subject={"reference": "Patient/q"})]}

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_budget_counts_helpers_and_never_dispatches_denied(self):
        calls = []
        budget = CallBudget(lambda *a, **k: calls.append(a) or outcome(200, "ok"), 2)
        budget("GET", "/Patient/p")
        budget("GET", "/Patient/p", actor="medguard")
        self.assertEqual(budget("POST", "/ServiceRequest", {})["status"], 429)
        self.assertEqual((len(calls), budget.used, budget.denied), (2, 2, 1))

    def test_operation_target_not_consumed_by_guard(self):
        proxy = StressProxy(self.store, self.task, "transient_503", {"placement": "agent_target"})
        self.assertEqual(proxy.call("GET", "/Patient/p", actor="medguard")["status"], 200)
        self.assertEqual(proxy.call("GET", "/Observation?patient=p&code=K", actor="medguard")["status"], 200)
        self.assertEqual(proxy.triggered, 0)
        self.assertEqual(proxy.call("GET", "/Observation?patient=p&code=K")["status"], 503)
        self.assertEqual(proxy.triggered, 1)

    def test_repeated_and_compound_counts(self):
        proxy = StressProxy(self.store, self.task, "transient_503+stale", {"repeat": 2})
        for _ in range(4):
            proxy.call("GET", "/Observation?patient=p&code=K")
        self.assertEqual(proxy.counts, {"transient_503": 2, "stale": 2})
        self.assertEqual(proxy.triggered, 4)

    def test_matching_public_timeouts_and_actual_commit(self):
        responses = []
        for hazard, n in (("commit_timeout", 1), ("no_commit_timeout", 0)):
            self.store.reset()
            proxy = StressProxy(self.store, self.task, hazard)
            responses.append(proxy.call("POST", "/ServiceRequest", service("p", "4548-4")))
            self.assertEqual(len([r for r in self.store.snapshot() if r["resourceType"] == "ServiceRequest"]), n)
        self.assertEqual(*responses)

    def test_guard_no_write_and_no_retry(self):
        for policy, hazard, count in (("medguard", "commit_timeout", 1), ("no_write", "commit_timeout", 2),
                                      ("no_retry", "no_commit_timeout", 0)):
            self.store.reset()
            proxy = StressProxy(self.store, self.task, hazard)
            FactorClient(proxy.call, policy).call("POST", "/ServiceRequest", service("p", "4548-4"))
            self.assertEqual(len([r for r in self.store.snapshot() if r["resourceType"] == "ServiceRequest"]), count)

    def test_evidence_only_does_not_claim_commit_or_retry(self):
        proxy = StressProxy(self.store, self.task, "commit_timeout")
        result = FactorClient(proxy.call, "evidence_only").call("POST", "/ServiceRequest", service("p", "4548-4"))
        self.assertEqual(result["status"], 504)
        self.assertEqual(len(result["verification_evidence"]["equivalent_records_after"]), 1)
        self.assertEqual(sum(t["method"] == "POST" for t in proxy.trace), 1)

    def test_delay_and_concurrency_are_not_hidden(self):
        for stress in ({"visibility_delay": 1}, {"concurrent": True}):
            self.store.reset()
            proxy = StressProxy(self.store, self.task, "commit_timeout", stress)
            FactorClient(proxy.call, "medguard").call("POST", "/ServiceRequest", service("p", "4548-4"))
            self.assertEqual(len([r for r in self.store.snapshot() if r["resourceType"] == "ServiceRequest"]), 2)
            if stress.get("concurrent"):
                self.assertEqual(len(proxy.external_ids), 1)
            else:
                self.assertTrue(any(t.get("visibility_masked") for t in proxy.trace))

    def test_read_component_removal(self):
        for policy, value in (("medguard", 3), ("no_read", 4)):
            proxy = StressProxy(self.store, self.task, "stale")
            response = FactorClient(proxy.call, policy).call("GET", "/Observation?patient=p&code=K&_count=1")
            self.assertEqual(response["body"]["entry"][0]["resource"]["valueQuantity"]["value"], value)

    def test_v2_metadata_describes_active_interface_between_faults(self):
        self.task["hazards"].append("interface_drift")
        proxy = InterfaceConsistentProxy(self.store, self.task, "interface_drift")
        for _ in range(2):
            response = proxy.call("GET", "/metadata")
            names = [p["name"] for p in response["body"]["rest"][0]["resource"][0]["searchParam"]]
            self.assertNotIn("patient", names)
            self.assertIn("subject", names)
            self.assertEqual(proxy.call("GET", "/Observation?patient=p&code=K")["status"], 400)
        self.assertEqual(proxy.call("GET", "/Observation?subject=p&code=K")["status"], 200)


if __name__ == "__main__":
    unittest.main()
