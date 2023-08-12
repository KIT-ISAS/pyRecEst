import unittest

import numpy as np
from pyrecest.distributions.circle.wrapped_laplace_distribution import (
    WrappedLaplaceDistribution,
)


class WrappedLaplaceDistributionTest(unittest.TestCase):
    def setUp(self):
        self.lambda_ = 2
        self.kappa = 1.3
        self.wl = WrappedLaplaceDistribution(self.lambda_, self.kappa)

    def test_pdf(self):
        def laplace(x):
            return (
                self.lambda_
                / (1 / self.kappa + self.kappa)
                * np.exp(
                    -(
                        abs(x)
                        * self.lambda_
                        * (self.kappa if x >= 0 else 1 / self.kappa)
                    )
                )
            )

        def pdftemp(x):
            return sum(laplace(z) for z in x + 2 * np.pi * np.arange(-20, 21))

        for x in [0, 1, 2, 3, 4]:
            np.testing.assert_allclose(self.wl.pdf(x), pdftemp(x), rtol=1e-10)

    def test_integral(self):
        np.testing.assert_allclose(self.wl.integrate(), 1, rtol=1e-10)
        np.testing.assert_allclose(self.wl.integrate_numerically(), 1, rtol=1e-10)
        np.testing.assert_allclose(
            self.wl.integrate([0, np.pi]) + self.wl.integrate([np.pi, 2 * np.pi]),
            1,
            rtol=1e-10,
        )

    def test_angular_moments(self):
        for i in range(1, 4):
            np.testing.assert_allclose(
                self.wl.trigonometric_moment(i),
                self.wl.trigonometric_moment_numerical(i),
                rtol=1e-10,
            )

    def test_periodicity(self):
        np.testing.assert_allclose(
            self.wl.pdf(np.linspace(-2 * np.pi, 0, 100)),
            self.wl.pdf(np.linspace(0, 2 * np.pi, 100)),
            rtol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
