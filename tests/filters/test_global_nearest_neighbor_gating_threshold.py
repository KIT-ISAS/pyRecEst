import unittest

from scipy.stats import chi2

from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


class GlobalNearestNeighborGatingThresholdTest(unittest.TestCase):
    def test_default_gate_is_derived_from_measurement_dimension(self):
        tracker = GlobalNearestNeighbor()

        self.assertIsNone(tracker.association_param["gating_distance_threshold"])
        self.assertAlmostEqual(
            tracker._get_gating_distance_threshold(2),
            chi2.ppf(0.999, 2),
        )
        self.assertAlmostEqual(
            tracker._get_gating_distance_threshold(3),
            chi2.ppf(0.999, 3),
        )

    def test_unsquared_gate_uses_square_root_chi_square_quantile(self):
        tracker = GlobalNearestNeighbor(association_param={"square_dist": False})

        self.assertAlmostEqual(
            tracker._get_gating_distance_threshold(2),
            chi2.ppf(0.999, 2) ** 0.5,
        )

    def test_explicit_gate_threshold_is_preserved(self):
        tracker = GlobalNearestNeighbor(
            association_param={"gating_distance_threshold": 7.5}
        )

        self.assertEqual(tracker._get_gating_distance_threshold(3), 7.5)


if __name__ == "__main__":
    unittest.main()
