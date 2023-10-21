from math import pi
from pyrecest.backend import arange
from pyrecest.backend import allclose
from pyrecest.backend import all
from pyrecest.backend import array
import pyrecest.backend
import unittest


from pyrecest.distributions import VonMisesDistribution, WrappedNormalDistribution


class AbstractCircularDistributionTest(unittest.TestCase):
    def setUp(self):
        self.distributions = [
            WrappedNormalDistribution(array(2.0), array(0.7)),
            VonMisesDistribution(array(6.0), array(1.2)),
        ]

    @unittest.skipIf(pyrecest.backend.__name__ == 'pyrecest.pytorch', reason="Not supported on PyTorch backend")
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

    def test_angular_moment_numerical(self):
        """Tests if the numerical computation of angular moment matches the actual moment."""
        moments = arange(4)

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

    def test_integral_numerical(self):
        """Tests if the numerical computation of integral matches the actual integral."""
        intervals = [
            (2, 2),
            (2, 3),
            (5, 4),
            (0, 4 * pi),
            (-pi, pi),
            (0, 4 * pi),
            (-3 * pi, 3 * pi),
            (-1, 20),
            (12, -3),
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