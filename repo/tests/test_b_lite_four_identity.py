import unittest
from medtoolbench.supplement.b_lite_four_identity import canonical_uuid, same_device


class GpuIdentityTests(unittest.TestCase):
    def test_prefix_is_display_only(self):
        value = "19c621d5-3f8a-78d9-c267-6ef556f382e9"
        self.assertTrue(same_device(value, "GPU-" + value))
        self.assertEqual(canonical_uuid("GPU-" + value), value)

    def test_different_physical_devices_rejected(self):
        self.assertFalse(same_device("19c621d5-3f8a-78d9-c267-6ef556f382e9",
                                    "GPU-a9219a96-eaee-9b74-ab2e-9e8eb68990b2"))

    def test_unrecognized_ids_fail_closed(self):
        for value in ("GPU-0", "0", "MIG-19c621d5-3f8a-78d9-c267-6ef556f382e9", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_uuid(value)


if __name__ == "__main__":
    unittest.main()
