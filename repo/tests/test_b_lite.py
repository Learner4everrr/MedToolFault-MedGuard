import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medtoolbench.supplement.b_lite import (
    select_questions, metadata_for, make_jobs, patient_effect, touched_patients,
    MODELS, HAZARDS, POLICIES, SETTINGS,
)
from medtoolbench.supplement.b_lite_execution import (
    eligible, inventory, model_gate, load, IntegrityError,
)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class BLiteTests(unittest.TestCase):
    def metadata(self):
        return [{"id": f"q{p}-{t}", "patient": f"p{p}", "template": f"t{t}",
                 "history_records": (p + 1) * 1000, "history_bin": p // 2}
                for p in range(6) for t in range(5)]

    def test_selection_covers_all_patients_templates_and_long_histories(self):
        rows = select_questions(self.metadata(), 12)
        self.assertEqual(len(rows), 12)
        for key in ("patient", "template", "history_bin"):
            self.assertEqual({r[key] for r in rows}, {r[key] for r in self.metadata()})
        self.assertIn(6000, {r["history_records"] for r in rows})

    def test_selection_reproducible_and_order_independent(self):
        rows = self.metadata()
        self.assertEqual(select_questions(rows, 12), select_questions(list(reversed(rows)), 12))
        self.assertNotEqual(select_questions(rows, 12, 1), select_questions(rows, 12, 2))

    def test_selection_forbids_label_and_outcome_fields(self):
        for field in ("success", "answer", "true_answer", "stop", "oom", "tokens"):
            rows = self.metadata()
            rows[0][field] = 0
            with self.assertRaises(ValueError):
                select_questions(rows, 12)

    def test_selection_rejects_duplicate_or_insufficient_population(self):
        for rows, count in ((self.metadata() * 2, 12), (self.metadata(), 31),
                            (self.metadata(), 1), (self.metadata(), 0)):
            with self.assertRaises(ValueError):
                select_questions(rows, count)

    def test_metadata_projection_does_not_expose_answers(self):
        q = {"question_id": "q", "patient_fhir_id": "p", "template": "t",
             "true_answer": "SECRET", "sql_query": "SECRET_SQL"}
        out = metadata_for([q], {"p": {"resources": 100}})
        self.assertNotIn("SECRET", str(out))
        self.assertEqual(out[0]["history_bin"], 0)

    def test_exposure_excludes_any_attempt_of_any_model_not_only_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            questions = [{"question_id": str(i), "patient_fhir_id": f"p{i}"} for i in range(3)]
            jobs = [{"id": str(i), "model": f"model{i}", "stage": "test",
                     "episodes": [{"question": str(i)}]} for i in range(3)]
            plan = root / "plan.json"
            put(plan, {"questions": questions, "jobs": jobs})
            (root / "jobs/0/attempt1").mkdir(parents=True)
            put(root / "jobs/1/result.json", {"state": "failed", "success": False})
            touched, proof = touched_patients([plan])
            self.assertEqual(touched, {"p0", "p1"})
            self.assertEqual(len(proof), 2)

    def test_full_factorial_and_budgets_unchanged(self):
        qs = [{"question_id": str(i), "patient_fhir_id": f"p{i // 3}"} for i in range(7)]
        jobs = make_jobs(qs, "test")
        self.assertEqual(sum(len(j["episodes"]) for j in jobs), 7 * 36)
        for model in MODELS:
            entries = [e for j in jobs if j["model"] == model for e in j["episodes"]]
            self.assertEqual(len({e["key"] for e in entries}), 7 * 18)
            self.assertEqual({e["condition"] for e in entries}, set(POLICIES))
            self.assertEqual({e["hazard"] for e in entries}, set(HAZARDS))
        self.assertTrue(all(j["settings"] == SETTINGS for j in jobs))
        self.assertTrue(all(len(j["episodes"]) <= 36 for j in jobs))

    def test_patient_bootstrap_is_not_episode_weighted(self):
        result = patient_effect({"a": [1] * 20, "b": [-1]}, 1000)
        self.assertEqual(result["difference"], 0)
        self.assertEqual(result["patients"], 2)
        self.assertEqual(result, patient_effect({"b": [-1], "a": [1] * 20}, 1000))

    def test_empty_conditional_stratum_is_explicit(self):
        result = patient_effect({})
        self.assertIsNone(result["difference"])
        self.assertIsNone(result["ci95"])

    def test_failed_job_is_not_requeued_and_other_model_still_eligible(self):
        with tempfile.TemporaryDirectory() as d:
            control = Path(d)
            jobs = [{"id": "q-test", "model": MODELS[0], "stage": "test"},
                    {"id": "l-test", "model": MODELS[1], "stage": "test"}]
            for m in MODELS:
                put(control / "preflight" / m / "result.json", {"state": "complete"})
                put(control / "preflight" / m / "complete.json", {"all_pass": True})
            put(control / "jobs/q-test/result.json", {"state": "quarantined"})
            with patch("medtoolbench.supplement.b_lite_execution.model_gate", return_value="passed"):
                choices = eligible(control, {"jobs": jobs}, set(), {})
                self.assertEqual([j["id"] for j in choices], ["l-test"])
                self.assertEqual(eligible(control, {"jobs": jobs}, {"l-test"}, {}), [])

    def test_formal_waits_for_qualification(self):
        with tempfile.TemporaryDirectory() as d:
            control = Path(d)
            jobs = [{"id": "dev", "model": MODELS[0], "stage": "qualification"},
                    {"id": "test", "model": MODELS[0], "stage": "test"}]
            for m in MODELS:
                put(control / "preflight" / m / "result.json", {"state": "complete"})
                put(control / "preflight" / m / "complete.json", {"all_pass": True})
            with patch("medtoolbench.supplement.b_lite_execution.model_gate", return_value="waiting"):
                self.assertEqual([j["id"] for j in eligible(control, {"jobs": jobs}, set(), {})], ["dev"])

    def test_preflight_cannot_be_launched_twice(self):
        with tempfile.TemporaryDirectory() as d:
            choices = eligible(Path(d), {"jobs": []}, {"preflight--" + MODELS[0]}, {MODELS[1]: "blocked"})
            self.assertEqual(choices, [])

    def test_failed_and_missing_shards_stay_in_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jobs = [{"id": str(i)} for i in range(3)]
            put(root / "jobs/0/result.json", {"state": "complete"})
            put(root / "jobs/1/result.json", {"state": "quarantined"})
            done, failures, pending = inventory(root, {"jobs": jobs})
            self.assertEqual((len(done), len(failures), len(pending)), (1, 1, 1))

    def test_model_gate_rejects_wrong_stress_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            put(root / "preflight" / MODELS[0] / "complete.json", {"all_pass": True, "plan_hash": "wrong"})
            with self.assertRaises(IntegrityError):
                model_gate(root, {"plan_hash": "right"}, MODELS[0])

    def test_loader_rejects_silent_budget_batch_and_model_changes(self):
        base = {"systems": [{"name": m, "batch_size": 2} for m in MODELS],
                "jobs": [{"settings": dict(SETTINGS)}]}
        mutations = []
        changed = copy.deepcopy(base)
        changed["systems"][0]["batch_size"] = 1
        mutations.append(changed)
        changed = copy.deepcopy(base)
        changed["jobs"][0]["settings"]["max_turns"] = 4
        mutations.append(changed)
        changed = copy.deepcopy(base)
        changed["systems"][1]["name"] = "new-model"
        mutations.append(changed)
        for plan in mutations:
            with patch("medtoolbench.supplement.b_lite_execution.load_plan", return_value=plan):
                with self.assertRaises(IntegrityError):
                    load(Path("unused"))


if __name__ == "__main__":
    unittest.main()
