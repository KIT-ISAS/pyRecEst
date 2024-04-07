import unittest
# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import array, all
from math import pi
from pyrecest.distributions import GeneralizedKSineSkewedVonMisesDistribution


class TestGeneralizedKSineSkewedVonMisesDistribution(unittest.TestCase):

    def test_initialization(self):
        """Test initialization with valid and invalid parameters."""
        # Test with valid parameters
        try:
            GeneralizedKSineSkewedVonMisesDistribution(mu=pi, kappa=1, lambda_=0.5, k=1, m=1)
        except NotImplementedError as e:
            self.fail(f"Initialization with valid parameters failed: {e}")

        # Test with invalid lambda_
        with self.assertRaises(AssertionError):
            GeneralizedKSineSkewedVonMisesDistribution(mu=pi, kappa=1, lambda_=1.5, k=1, m=1)

        # Test with invalid m
        with self.assertRaises(AssertionError):
            GeneralizedKSineSkewedVonMisesDistribution(mu=pi, kappa=1, lambda_=0.5, k=1, m=0)

    def test_pdf(self):
        """Test the pdf method for expected behavior."""
        dist = GeneralizedKSineSkewedVonMisesDistribution(mu=pi, kappa=1, lambda_=0.5, k=1, m=1)
        xa = pi / 2
        pdf_val = dist.pdf(xa)
        # This is more of a sanity check; a more comprehensive test would compare against a known result
        self.assertTrue(0 <= pdf_val <= 2, "PDF value out of expected range.")

    def test_shift(self):
        """Test the shift method modifies mu correctly."""
        dist = GeneralizedKSineSkewedVonMisesDistribution(mu=0, kappa=1, lambda_=0.5, k=1, m=1)
        new_dist = dist.shift(array(pi / 2))
        self.assertEqual(new_dist.mu, pi / 2, "Shift method did not update mu correctly.")


if __name__ == '__main__':
    unittest.main()
