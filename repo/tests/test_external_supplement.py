import unittest

from medtoolbench.supplement.code_worker import checked_tree, evaluate
from medtoolbench.supplement.external_runner import correct, parse
from medtoolbench.supplement.external_tools import ExternalTools, Retrieval, messages


class ExternalSupplementTests(unittest.TestCase):
    def setUp(self):
        self.lab = {"resourceType": "Observation", "id": "o1", "status": "final", "subject": {"reference": "Patient/p"},
            "effectiveDateTime": "2133-01-01T09:00:00Z", "valueQuantity": {"value": 3}, "code": {"coding": [{"code": "1", "display": "potassium"}]}}
        self.data = {"Patient": [{"resourceType": "Patient", "id": "p"}], "Encounter": [],
            "Observation": [self.lab, {**self.lab, "id": "o2", "effectiveDateTime": "2133-01-02T09:00:00Z"}]}
        self.other = {"Observation": [{**self.lab, "subject": {"reference": "Patient/q"}}]}

    def test_code_cannot_open_files_network_or_private_names(self):
        for code in ("import os", "import socket", "from pathlib import Path", "x = (1).__class__", "x = __builtins__", "from datetime import __dict__"):
            with self.assertRaises(ValueError):
                checked_tree(code)
        with self.assertRaises(NameError):
            evaluate("open('/etc/passwd')", {})
        with self.assertRaises(NameError):
            evaluate("eval('1+1')", {})

    def test_fhir_calculation_and_timestamp(self):
        result = evaluate("from datetime import datetime\nxs = retrieved_resources['Observation']\nresult = [[sum(x['valueQuantity']['value'] for x in xs)]]\nprint(datetime.fromisoformat(xs[0]['effectiveDateTime'].replace('Z','+00:00')).year)", self.data)
        self.assertEqual(result["result"], [[6]])
        self.assertEqual(result["stdout"].strip(), "2133")

    def test_reference_answer_never_enters_public_prompt(self):
        q = {"question": "question", "assumption": "time", "patient_fhir_id": "p", "true_answer": "SECRET", "sql_query": "SECRET_SQL"}
        self.assertNotIn("SECRET", str(messages(q, "medguard")))

    def test_supported_datetime_formatting(self):
        result = evaluate("from datetime import datetime\nresult=datetime(2020,1,2).strftime('%Y-%m-%d')", {})
        self.assertEqual(result["result"], "2020-01-02")

    def test_every_fault_has_public_authoritative_recovery(self):
        for hazard in ("clean", "interface_drift", "transient_503", "partial_history", "patient_mismatch", "provenance_conflict"):
            retrieval = Retrieval(self.data, "p", hazard, self.other)
            tools = ExternalTools(retrieval, "medguard")
            response = tools.call({"tool": "get_resources", "resource_type": "Observation", "patient_fhir_id": "p"})
            self.assertEqual(response["status"], 200, hazard)
            self.assertEqual(tools.retrieved["Observation"], self.data["Observation"], hazard)

    def test_partial_and_identity_faults_are_real_not_only_status(self):
        for hazard in ("partial_history", "patient_mismatch", "provenance_conflict"):
            tools = ExternalTools(Retrieval(self.data, "p", hazard, self.other), "vanilla")
            tools.call({"tool": "get_resources", "resource_type": "Observation", "patient_fhir_id": "p"})
            self.assertNotEqual(tools.retrieved["Observation"], self.data["Observation"])
            self.assertEqual(tools.retrieval.triggered, 1)

    def test_external_final_dispatch_and_grading(self):
        self.assertEqual(parse('{"tool":"metadata"}{"final":{"rows":[]}}'), {"tool": "metadata"})
        with self.assertRaises(ValueError):
            parse('{"final":{"rows":[]},"tool":"metadata"}')
        self.assertTrue(correct({"rows": [[1]]}, [[1.0]]))
        self.assertFalse(correct({"rows": []}, [[None]]))
        self.assertFalse(correct({"rows": ["hello"]}, [["hello"]]))


if __name__ == "__main__":
    unittest.main()
