import math
import unittest

import numpy.testing as npt
from scipy.special import ive

from pyrecest.backend import array
from pyrecest.distributions.circle.von_mises_distribution import VonMisesDistribution
from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import (
    VonMisesFisherDistribution,
)


class TestStableDirectionalDensities(unittest.TestCase):
    def test_von_mises_pdf_is_finite_for_large_kappa(self):
        mu = 0.3
        kappa = 1000.0
        dist = VonMisesDistribution(mu, kappa)

        # Accessing the ordinary normalizer may overflow because the true
        # normalizer is larger than finite double precision. It must not poison
        # subsequent density evaluation.
        _ = dist.norm_const
        mode_pdf = float(dist.pdf(array([mu]))[0])
        expected_log_pdf = kappa - (
            math.log(2.0 * math.pi) + math.log(float(ive(0, kappa))) + kappa
        )

        self.assertTrue(math.isfinite(mode_pdf))
        self.assertGreater(mode_pdf, 0.0)
        npt.assert_allclose(mode_pdf, math.exp(expected_log_pdf), rtol=1e-12)

        entropy = float(dist.entropy())
        self.assertTrue(math.isfinite(entropy))

    def test_von_mises_fisher_pdf_is_finite_for_large_kappa(self):
        kappa = 1000.0
        mu = array([1.0, 0.0, 0.0])
        dist = VonMisesFisherDistribution(mu, kappa)

        mode_pdf = float(dist.pdf(array([[1.0, 0.0, 0.0]]))[0])
        nu = 0.5
        expected_log_c = (
            nu * math.log(kappa)
            - 1.5 * math.log(2.0 * math.pi)
            - (math.log(float(ive(nu, kappa))) + kappa)
        )
        expected_log_pdf = expected_log_c + kappa

        self.assertTrue(math.isfinite(float(dist.log_C)))
        self.assertTrue(math.isfinite(mode_pdf))
        self.assertGreater(mode_pdf, 0.0)
        npt.assert_allclose(mode_pdf, math.exp(expected_log_pdf), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
