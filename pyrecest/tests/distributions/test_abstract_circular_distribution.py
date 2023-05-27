import unittest

import numpy as np
from pyrecest.distributions import VonMisesDistribution, WrappedNormalDistribution


class AbstractCircularDistributionTest(unittest.TestCase):
    def setUp(self):
        self.distributions = [
            WrappedNormalDistribution(2, 0.7),
            VonMisesDistribution(6, 1.2),
        ]

    def test_cdf_numerical(self):
        """Tests if the numerical computation of cdf matches the actual cdf."""
        x = np.arange(0, 7)
        starting_point = 2.1

        for dist in self.distributions:
            with self.subTest(distribution=dist):
                self.assertTrue(
                    np.allclose(dist.cdf_numerical(x), dist.cdf(x), rtol=1e-10)
                )
                self.assertTrue(
                    np.allclose(
                        dist.cdf_numerical(x, starting_point),
                        dist.cdf(x, starting_point),
                        rtol=1e-10,
                    )
                )

    def test_angular_moment_numerical(self):
        """Tests if the numerical computation of angular moment matches the actual moment."""
        moments = np.arange(4)

        for dist in self.distributions:
            for moment in moments:
                with self.subTest(distribution=dist, moment=moment):
                    self.assertTrue(
                        np.allclose(
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
            (0, 4 * np.pi),
            (-np.pi, np.pi),
            (0, 4 * np.pi),
            (-3 * np.pi, 3 * np.pi),
            (-1, 20),
            (12, -3),
        ]

        for dist in self.distributions:
            for interval in intervals:
                with self.subTest(distribution=dist, interval=interval):
                    self.assertTrue(
                        np.allclose(
                            dist.integrate_numerically(interval),
                            dist.integrate(interval),
                            rtol=1e-10,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
