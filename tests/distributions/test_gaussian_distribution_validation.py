import unittest

from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution


class TestGaussianDistributionValidation(unittest.TestCase):
    def test_constructor_rejects_nonfinite_mean_by_default(self):
        covariance = eye(1)

        for bad_mean in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad_mean=bad_mean):
                with self.assertRaisesRegex(ValueError, "mu.*finite"):
                    GaussianDistribution(array([bad_mean]), covariance)

    def test_constructor_rejects_nonfinite_covariance_by_default(self):
        mean = array([0.0])

        for bad_covariance in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad_covariance=bad_covariance):
                with self.assertRaisesRegex(ValueError, "C.*finite"):
                    GaussianDistribution(mean, array([[bad_covariance]]))


if __name__ == "__main__":
    unittest.main()
