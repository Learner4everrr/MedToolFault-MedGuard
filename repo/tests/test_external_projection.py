import sqlite3
import unittest

from medtoolbench.research.fhir_projection import identifier, local_clock, project
from medtoolbench.research.sql_reference import authorizer, equivalent_rows


class ExternalProjectionTests(unittest.TestCase):
    def test_local_sql_clock_is_not_utc(self):
        self.assertEqual(local_clock("2133-12-31T02:00:00-05:00"), "2133-12-31 02:00:00")
        self.assertIsNone(local_clock(None))
        with self.assertRaises(ValueError):
            local_clock("today")

    def test_identifier_ambiguity_is_rejected(self):
        r = {"identifier": [{"system": "x/patient", "value": "1"}, {"system": "x/patient", "value": "2"}]}
        with self.assertRaises(ValueError):
            identifier(r, "patient")

    def test_fhir_only_projection_preserves_encounter_and_null_values(self):
        records = {
            "Patient": [{"id": "p", "identifier": [{"system": "x/patient", "value": "100"}]}],
            "Encounter": [
                {"id": "icu", "subject": {"reference": "Patient/p"}, "partOf": {"reference": "Encounter/h"},
                 "identifier": [{"system": "x/encounter-icu", "value": "300"}]},
                {"id": "h", "subject": {"reference": "Patient/p"},
                 "identifier": [{"system": "x/encounter-hosp", "value": "200"}],
                 "period": {"start": "2133-12-01T04:00:00-05:00"}}],
            "Observation": [
                {"id": "o", "meta": {"profile": ["x/mimic-observation-chartevents"]},
                 "subject": {"reference": "Patient/p"}, "encounter": {"reference": "Encounter/icu"},
                 "code": {"coding": [{"system": "x/mimic-chartevents-d-items", "code": "123", "display": "RR"}]},
                 "effectiveDateTime": "2133-12-31T02:00:00-05:00", "valueQuantity": {"value": 21}},
                {"id": "l", "meta": {"profile": ["x/mimic-observation-labevents"]},
                 "subject": {"reference": "Patient/p"},
                 "code": {"coding": [{"system": "x/mimic-d-labitems", "code": "456", "display": "Specimen"}]},
                 "effectiveDateTime": "2133-12-31T02:00:00-05:00", "valueString": "ART."}]
        }
        db = sqlite3.connect(":memory:")
        counts = project(db, lambda kind: iter(records[kind]))
        self.assertEqual(counts["chartevents"], 1)
        self.assertEqual(db.execute("SELECT subject_id,hadm_id,stay_id,valuenum FROM chartevents").fetchone(), (100, 200, 300, 21))
        self.assertEqual(db.execute("SELECT hadm_id,valuenum FROM labevents").fetchone(), (None, None))
        self.assertEqual(db.execute("SELECT label,linksto FROM d_items").fetchone(), ("RR", "chartevents"))
        db.close()

    def test_query_authorizer_rejects_writes_and_external_access(self):
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE sample(value INTEGER)")
        db.execute("INSERT INTO sample VALUES(1)")
        db.commit()
        db.set_authorizer(authorizer)
        self.assertEqual(db.execute("SELECT sum(value) FROM sample").fetchone(), (1,))
        for query in ("DELETE FROM sample", "DROP TABLE sample", "ATTACH DATABASE ':memory:' AS hidden", "PRAGMA database_list", "SELECT load_extension('x')"):
            with self.assertRaises(sqlite3.DatabaseError):
                db.execute(query)
        db.close()

    def test_answer_comparison_is_multiset_not_set(self):
        self.assertTrue(equivalent_rows([[1.0000001], [" YES "]], [["yes"], [1]]))
        self.assertFalse(equivalent_rows([[1], [1]], [[1], [2]]))
        self.assertFalse(equivalent_rows([[None]], [[0]]))


if __name__ == "__main__":
    unittest.main()
