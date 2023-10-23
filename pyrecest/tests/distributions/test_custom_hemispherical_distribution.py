import unittest
import warnings

from pyrecest.backend import allclose, array, eye, linalg, ndim, random
from pyrecest.distributions import (
    BinghamDistribution,
    CustomHemisphericalDistribution,
    VonMisesFisherDistribution,
)


class CustomHemisphericalDistributionTest(unittest.TestCase):
    def setUp(self):
        self.M = eye(3)
        self.Z = array([-2.0, -0.5, 0.0])
        self.bingham_distribution = BinghamDistribution(self.Z, self.M)
        self.custom_hemispherical_distribution = (
            CustomHemisphericalDistribution.from_distribution(self.bingham_distribution)
        )

    def test_simple_distribution_2D(self):
        """Test that pdf function returns the correct size and values for given points."""
        p = self.custom_hemispherical_distribution.pdf(array([1.0, 0.0, 0.0]))
        self.assertEqual(ndim(p), 0, "PDF size mismatch.")

        random.seed(10)
        points = random.normal(0.0, 1.0, (100, 3))
        points = points[points[:, 2] >= 0.0, :]
        points /= linalg.norm(points, axis=1).reshape(-1, 1)

        self.assertTrue(
            allclose(
                self.custom_hemispherical_distribution.pdf(points),
                2.0 * self.bingham_distribution.pdf(points),
                atol=1e-5,
            ),
            "PDF values do not match.",
        )

    def test_integrate_bingham_s2(self):
        """Test that the distribution integrates to 1."""
        self.custom_hemispherical_distribution.pdf(array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(
            self.custom_hemispherical_distribution.integrate_numerically(),
            1,
            delta=1e-4,
            msg="Integration value mismatch.",
        )

    def test_warning_asymmetric(self):
        """Test that creating a custom distribution based on a full hypersphere distribution raises a warning."""
        vmf = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 10.0)
        expected_warning_message = (
            "You are creating a CustomHyperhemispherical distribution based on a distribution on the full hypersphere. "
            + "Using numerical integration to calculate the normalization constant."
        )

        with warnings.catch_warnings(record=True) as warning_list:
            CustomHemisphericalDistribution.from_distribution(vmf)

            # Check the warning message
            self.assertEqual(
                str(warning_list[-1].message),
                expected_warning_message,
                "Warning message mismatch.",
            )


if __name__ == "__main__":
    unittest.main()
