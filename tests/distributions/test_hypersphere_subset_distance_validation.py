import unittest

from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)


class HypersphereSubsetDistanceValidationTest(unittest.TestCase):
    def test_hellinger_distance_rejects_dimension_mismatch(self):
        dist = HypersphericalUniformDistribution(2)
        other = HypersphericalUniformDistribution(3)

        with self.assertRaisesRegex(ValueError, "different number of dimensions"):
            dist.hellinger_distance_numerical(other)

    def test_total_variation_distance_rejects_dimension_mismatch(self):
        dist = HypersphericalUniformDistribution(2)
        other = HypersphericalUniformDistribution(3)

        with self.assertRaisesRegex(ValueError, "different number of dimensions"):
            dist.total_variation_distance_numerical(other)


if __name__ == "__main__":
    unittest.main()
