import unittest
import numpy as np
from pyrecest.distributions import GeneralizedKSineSkewedVonMisesDistribution  # Assuming your class is in 'your_module.py'


class TestGeneralizedKSineSkewedVonMisesDistribution(unittest.TestCase):

    def test_initialization(self):
        """Test initialization with valid and invalid parameters."""
        # Test with valid parameters
        try:
            GeneralizedKSineSkewedVonMisesDistribution(mu=np.pi, kappa=1, lambda_=0.5, k=1, m=1)
        except NotImplementedError as e:
            self.fail(f"Initialization with valid parameters failed: {e}")

        # Test with invalid lambda_
        with self.assertRaises(AssertionError):
            GeneralizedKSineSkewedVonMisesDistribution(mu=np.pi, kappa=1, lambda_=1.5, k=1, m=1)

        # Test with invalid m
        with self.assertRaises(AssertionError):
            GeneralizedKSineSkewedVonMisesDistribution(mu=np.pi, kappa=1, lambda_=0.5, k=1, m=0)

    def test_pdf(self):
        """Test the pdf method for expected behavior."""
        dist = GeneralizedKSineSkewedVonMisesDistribution(mu=np.pi, kappa=1, lambda_=0.5, k=1, m=1)
        xa = np.pi / 2
        pdf_val = dist.pdf(xa)
        # This is more of a sanity check; a more comprehensive test would compare against a known result
        self.assertTrue(0 <= pdf_val <= 2, "PDF value out of expected range.")

    def test_shift(self):
        """Test the shift method modifies mu correctly."""
        dist = GeneralizedKSineSkewedVonMisesDistribution(mu=0, kappa=1, lambda_=0.5, k=1, m=1)
        new_dist = dist.shift(np.pi / 2)
        self.assertEqual(new_dist.mu, np.pi / 2, "Shift method did not update mu correctly.")


if __name__ == '__main__':
    unittest.main()
