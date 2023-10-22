from math import pi
from pyrecest.backend import linspace
from pyrecest.backend import exp
from pyrecest.backend import arange
from pyrecest.backend import array
import pyrecest.backend
import unittest
import numpy.testing as npt

from pyrecest.distributions.circle.wrapped_laplace_distribution import (
    WrappedLaplaceDistribution,
)


class WrappedLaplaceDistributionTest(unittest.TestCase):
    def setUp(self):
        self.lambda_ = array(2.0)
        self.kappa = array(1.3)
        self.wl = WrappedLaplaceDistribution(self.lambda_, self.kappa)

    def test_pdf(self):
        def laplace(x):
            return (
                self.lambda_
                / (1 / self.kappa + self.kappa)
                * exp(
                    -(
                        abs(x)
                        * self.lambda_
                        * (self.kappa if x >= 0 else 1 / self.kappa)
                    )
                )
            )

        def pdftemp(x):
            return sum(laplace(z) for z in x + 2.0 * pi * arange(-20, 21))

        for x in [0.0, 1.0, 2.0, 3.0, 4.0]:
            npt.assert_allclose(self.wl.pdf(x), pdftemp(x), rtol=1e-10)

    @unittest.skipIf(pyrecest.backend.__name__ == 'pyrecest.pytorch', reason="Not supported on PyTorch backend")
    def test_integral(self):
        npt.assert_allclose(self.wl.integrate(), 1.0, rtol=1e-10)
        npt.assert_allclose(self.wl.integrate_numerically(), 1.0, rtol=1e-10)
        npt.assert_allclose(
            self.wl.integrate(array([0.0, pi])) + self.wl.integrate(array([pi, 2.0 * pi])),
            1,
            rtol=1e-10,
        )

    def test_angular_moments(self):
        for i in range(1, 4):
            npt.assert_allclose(
                self.wl.trigonometric_moment(i),
                self.wl.trigonometric_moment_numerical(i),
                rtol=1e-10,
            )

    def test_periodicity(self):
        npt.assert_allclose(
            self.wl.pdf(linspace(-2 * pi, 0, 100)),
            self.wl.pdf(linspace(0, 2 * pi, 100)),
            rtol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()