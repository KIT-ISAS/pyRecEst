import unittest
from math import pi

from pyrecest.backend import array
from pyrecest.distributions.circle.sine_skewed_distributions import (
    GeneralizedKSineSkewedVonMisesDistribution,
    SineSkewedWrappedCauchyDistribution,
    SineSkewedWrappedNormalDistribution,
)


class TestGeneralizedKSineSkewedVonMisesDistribution(unittest.TestCase):
    def test_initialization(self):
        """Test initialization with valid and invalid parameters."""
        # Test with valid parameters
        try:
            GeneralizedKSineSkewedVonMisesDistribution(
                mu=pi, kappa=1, lambda_=0.5, k=1, m=1
            )
        except NotImplementedError as e:
            self.fail(f"Initialization with valid parameters failed: {e}")

        # Test with invalid lambda_
        with self.assertRaises(AssertionError):
            GeneralizedKSineSkewedVonMisesDistribution(
                mu=pi, kappa=1, lambda_=1.5, k=1, m=1
            )

        # Test with invalid m
        with self.assertRaises(AssertionError):
            GeneralizedKSineSkewedVonMisesDistribution(
                mu=pi, kappa=1, lambda_=0.5, k=1, m=0
            )

    def test_pdf(self):
        """Test the pdf method for expected behavior."""
        dist = GeneralizedKSineSkewedVonMisesDistribution(
            mu=pi, kappa=1, lambda_=0.5, k=1, m=1
        )
        xa = pi / 2
        pdf_val = dist.pdf(xa)
        # This is more of a sanity check; a more comprehensive test would compare against a known result
        self.assertTrue(0 <= pdf_val <= 2, "PDF value out of expected range.")

    def test_shift(self):
        """Test the shift method modifies mu correctly."""
        dist = GeneralizedKSineSkewedVonMisesDistribution(
            mu=0, kappa=1, lambda_=0.5, k=1, m=1
        )
        new_dist = dist.shift(array(pi / 2))
        self.assertEqual(
            new_dist.mu, pi / 2, "Shift method did not update mu correctly."
        )


def test_sine_skewed_wrapped_normal_initialization():
    mu = array(0.0)
    sigma = array(1.0)
    lambda_ = array(0.5)
    dist = SineSkewedWrappedNormalDistribution(mu, sigma, lambda_)
    assert dist.mu == mu
    assert dist.sigma == sigma
    assert dist.lambda_ == lambda_


def test_sine_skewed_wrapped_cauchy_initialization():
    mu = array(pi / 4)
    gamma = array(0.1)
    lambda_ = array(-0.5)
    dist = SineSkewedWrappedCauchyDistribution(mu, gamma, lambda_)
    assert dist.mu == mu
    assert dist.gamma == gamma
    assert dist.lambda_ == lambda_


def test_sine_skewed_wrapped_normal_pdf():
    mu = array(0.0)
    sigma = array(1.0)
    lambda_ = array(0.0)
    xs = array([0.0, pi / 2, pi, 3 * pi / 2])
    dist = SineSkewedWrappedNormalDistribution(mu, sigma, lambda_)
    pdf_values = dist.pdf(xs)
    assert len(pdf_values) == len(xs)
    assert all(pdf_values >= 0)  # PDF values should be non-negative


def test_sine_skewed_wrapped_cauchy_pdf():
    mu = array(pi / 4)
    gamma = array(0.1)
    lambda_ = array(0.0)
    xs = array([0.0, pi / 2, pi, 3 * pi / 2])
    dist = SineSkewedWrappedCauchyDistribution(mu, gamma, lambda_)
    pdf_values = dist.pdf(xs)
    assert len(pdf_values) == len(xs)
    assert all(pdf_values >= 0)  # PDF values should be non-negative


def test_sine_skewed_effect():
    mu = array(0.0)
    sigma = array(1.0)
    lambda_ = array(1.0)  # Max skew
    normal_dist = SineSkewedWrappedNormalDistribution(mu, sigma, 0)  # No skew
    skewed_dist = SineSkewedWrappedNormalDistribution(mu, sigma, lambda_)  # Max skew
    assert skewed_dist.pdf(mu + 0.1) > normal_dist.pdf(mu)
    assert skewed_dist.pdf(mu - 0.1) < normal_dist.pdf(mu)

    lambda_ = -1  # Negative skew
    skewed_dist = SineSkewedWrappedNormalDistribution(mu, sigma, lambda_)
    assert skewed_dist.pdf(mu + 0.1) < normal_dist.pdf(mu)
    assert skewed_dist.pdf(mu - 0.1) > normal_dist.pdf(mu)


if __name__ == "__main__":
    unittest.main()
