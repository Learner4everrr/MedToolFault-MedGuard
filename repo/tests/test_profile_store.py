import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from medtoolbench.research.fhir import HapiStore, normalized_state, state_key
from medtoolbench.supplement.profile_store import ProfilePreservingProcess, ProfilePreservingStore, import_payload


class MemoryStore(ProfilePreservingStore):
    def __init__(self, resources=(), drop_profile=False):
        super().__init__(SimpleNamespace(port=0, database_copy=None))
        self.resources = {state_key(r): copy.deepcopy(r) for r in resources}
        self.transactions = []
        self.drop_profile = drop_profile

    def snapshot(self):
        return sorted(copy.deepcopy(list(self.resources.values())), key=state_key)

    def transaction(self, entries):
        self.transactions.extend(copy.deepcopy(entries))
        for entry in entries:
            key = tuple(entry["request"]["url"].split("/"))
            if entry["request"]["method"] == "DELETE":
                self.resources.pop(key, None)
            else:
                resource = copy.deepcopy(entry["resource"])
                if self.drop_profile:
                    resource.pop("meta", None)
                self.resources[key] = resource


class ProfileStoreTests(unittest.TestCase):
    def setUp(self):
        self.row = {"resourceType": "Patient", "id": "p", "gender": "female", "meta": {
            "profile": ["urn:test:profile"], "security": [{"code": "R"}], "tag": [{"code": "clinical"}],
            "versionId": "123", "lastUpdated": "2020-01-01T00:00:00Z", "source": "old-server"}}

    def test_import_preserves_profiles_security_tags_without_mutating_source(self):
        original = copy.deepcopy(self.row)
        payload = import_payload(self.row)
        self.assertEqual(set(payload["meta"]), {"profile", "security", "tag"})
        self.assertEqual(payload["meta"]["profile"], self.row["meta"]["profile"])
        self.assertEqual(self.row, original)

    def test_volatile_metadata_only_matches_original_payload(self):
        row = {"resourceType": "Patient", "id": "p", "meta": {"versionId": "1", "source": "old"}}
        self.assertEqual(import_payload(row), {"resourceType": "Patient", "id": "p"})
        self.assertEqual(import_payload({"resourceType": "Patient", "id": "q"}), {"resourceType": "Patient", "id": "q"})

    def test_original_reset_fails_profile_fixture_but_corrected_reset_passes(self):
        with self.assertRaisesRegex(AssertionError, "differs"):
            HapiStore.reload(MemoryStore(), [self.row])
        store = MemoryStore()
        self.assertEqual(normalized_state(store.reload([self.row])), normalized_state([self.row]))
        store.reset()
        self.assertEqual(len(store.transactions), 1)

    def test_strict_check_still_rejects_dropped_metadata(self):
        with self.assertRaisesRegex(AssertionError, "differs"):
            MemoryStore(drop_profile=True).reload([self.row])

    def test_reset_removes_prior_writes_and_restores_changed_data(self):
        store = MemoryStore()
        store.reload([self.row])
        store.resources[("Patient", "p")]["gender"] = "male"
        store.resources[("ServiceRequest", "extra")] = {"resourceType": "ServiceRequest", "id": "extra"}
        store.reset()
        self.assertEqual(normalized_state(store.snapshot()), normalized_state([self.row]))

    def test_source_database_write_guard_is_retained(self):
        store = MemoryStore()
        store.owner.database_copy = "/source/copy"
        with self.assertRaisesRegex(RuntimeError, "Refusing"):
            store.reload([self.row])
        self.assertEqual(store.transactions, [])

    def test_process_changes_only_its_owned_store(self):
        from medtoolbench.research import fhir
        original = fhir.HapiStore
        process = ProfilePreservingProcess("/fake-image", "/fake-logs")
        process.port = 0
        with patch.object(fhir.HapiProcess, "start", return_value=object()):
            self.assertIsInstance(process.start(), ProfilePreservingStore)
        self.assertIs(fhir.HapiStore, original)


if __name__ == "__main__":
    unittest.main()
