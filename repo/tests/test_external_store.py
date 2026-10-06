import unittest
from urllib.parse import parse_qs, urlsplit

from medtoolbench.research.external_store import fetch_kind


class ExternalStoreTests(unittest.TestCase):
    def test_receipt_batches_cover_large_histories(self):
        class Store:
            calls = []
            def all_pages(self, path):
                self.calls.append(path)
                ids = parse_qs(urlsplit(path).query)["_id"][0].split(",")
                return [{"resourceType": "Observation", "id": i} for i in reversed(ids)]
        store = Store()
        ids = [str(i) for i in range(5501)]
        rows = fetch_kind(store, "Observation", ids)
        self.assertEqual(len(store.calls), 28)
        self.assertEqual({r["id"] for r in rows}, set(ids))

    def test_truncation_and_duplicate_receipts_fail_closed(self):
        class Store:
            def all_pages(self, path):
                return []
        with self.assertRaises(RuntimeError):
            fetch_kind(Store(), "Observation", ["a"])
        with self.assertRaises(ValueError):
            fetch_kind(Store(), "Observation", ["a", "a"])


if __name__ == "__main__":
    unittest.main()
