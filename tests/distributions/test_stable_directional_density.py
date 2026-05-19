import math
import unittest

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import VonMisesDistribution, VonMisesFisherDistribution


class StableDirectionalDensityTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="These checks rely on Python scalar conversion of backend values.",
    )
    def test_von_mises_high_kappa_pdf_and_entropy_are_finite(self):
        dist = VonMisesDistribution(array(0.3), array(1000.0))

        mode_log_density = float(dist.log_pdf(array([0.3]))[0])
        mode_density = float(dist.pdf(array([0.3]))[0])
        bessel_ratio = float(VonMisesDistribution.besselratio(0, array(1000.0)))
        entropy = float(dist.entropy())

        self.assertTrue(math.isfinite(mode_log_density))
        self.assertTrue(math.isfinite(mode_density))
        self.assertGreater(mode_density, 0.0)
        self.assertTrue(math.isfinite(bessel_ratio))
        self.assertLess(bessel_ratio, 1.0)
        self.assertTrue(math.isfinite(entropy))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="These checks rely on Python scalar conversion of backend values.",
    )
    def test_vmf_high_kappa_pdf_is_finite_at_mode(self):
        mean_direction = array([1.0, 0.0, 0.0])
        dist = VonMisesFisherDistribution(mean_direction, array(1000.0))

        mode_log_density = float(dist.log_pdf(mean_direction))
        mode_density = float(dist.pdf(mean_direction))

        self.assertTrue(math.isfinite(mode_log_density))
        self.assertTrue(math.isfinite(mode_density))
        self.assertGreater(mode_density, 0.0)


if __name__ == "__main__":
    unittest.main()
