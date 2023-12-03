import unittest
from math import pi

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, arange, array
from pyrecest.distributions import VonMisesDistribution, WrappedNormalDistribution


class AbstractCircularDistributionTest(unittest.TestCase):
    def setUp(self):
        self.distributions = [
            WrappedNormalDistribution(array(2.0), array(0.7)),
            VonMisesDistribution(array(6.0), array(1.2)),
        ]

    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.pytorch",
        reason="Not supported on PyTorch backend",
    )
    def test_cdf_numerical(self):
        """Tests if the numerical computation of cdf matches the actual cdf."""
        x = arange(0, 7)
        starting_point = 2.1

        for dist in self.distributions:
            with self.subTest(distribution=dist):
                self.assertTrue(
                    allclose(dist.cdf_numerical(x), dist.cdf(x), rtol=1e-10)
                )
                self.assertTrue(
                    allclose(
                        dist.cdf_numerical(x, starting_point),
                        dist.cdf(x, starting_point),
                        rtol=1e-10,
                    )
                )

    def test_angularmoment_numerical(self):
        """Tests if the numerical computation of angular moment matches the actual moment."""
        moments = arange(3)

        for dist in self.distributions:
            for moment in moments:
                with self.subTest(distribution=dist, moment=moment):
                    self.assertTrue(
                        allclose(
                            dist.trigonometric_moment(moment),
                            dist.trigonometric_moment_numerical(moment),
                            rtol=1e-10,
                        )
                    )

    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.pytorch",
        reason="Not supported on PyTorch backend",
    )
    def test_integral_numerical(self):
        """Tests if the numerical computation of integral matches the actual integral."""
        intervals = [
            (2.0, 2.0),
            (2.0, 3.0),
            (5.0, 4.0),
            (0.0, 4.0 * pi),
            (-pi, pi),
            (0.0, 4.0 * pi),
            (-3.0 * pi, 3.0 * pi),
            (-1.0, 20.0),
            (12.0, -3.0),
        ]

        for dist in self.distributions:
            for interval in intervals:
                with self.subTest(distribution=dist, interval=interval):
                    self.assertTrue(
                        allclose(
                            dist.integrate_numerically(array(interval)),
                            dist.integrate(array(interval)),
                            rtol=1e-10,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
