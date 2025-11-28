from typing import Union

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos, int32, int64, pi, sin, sqrt, zeros
from scipy.integrate import dblquad

from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class AbstractToroidalDistribution(AbstractHypertoroidalDistribution):
    def __init__(self):
        AbstractHypertoroidalDistribution.__init__(self, 2)

    def covariance_4D_numerical(self):
        m = self.mean_4D()

        def f(
            x: float,
            y: float,
            i: Union[int, int32, int64],
            j: Union[int, int32, int64],
        ) -> float:
            funcs = [
                lambda x, _: cos(x) - m[0],
                lambda x, _: sin(x) - m[1],
                lambda _, y: cos(y) - m[2],
                lambda _, y: sin(y) - m[3],
            ]
            return self.pdf(array([x, y])) * funcs[i](x, y) * funcs[j](x, y)

        C = zeros((4, 4))
        for i in range(4):
            for j in range(i, 4):
                C[i, j], _ = dblquad(f, 0, 2 * pi, 0, 2 * pi, args=(i, j))
                if i != j:
                    C[j, i] = C[i, j]

        return C

    def circular_correlation_jammalamadaka(self) -> float:
        rhoc = self.circular_correlation_jammalamadaka_numerical()
        return rhoc

    def circular_correlation_jammalamadaka_numerical(self) -> float:
        m = self.mean_direction()

        def fsinAsinB(x, y):
            return self.pdf(array([x, y])) * sin(x - m[0]) * sin(y - m[1])

        def fsinAsquared(x, y):
            return self.pdf(array([x, y])) * sin(x - m[0]) ** 2

        def fsinBsquared(x, y):
            return self.pdf(array([x, y])) * sin(y - m[1]) ** 2

        EsinAsinB, _ = dblquad(fsinAsinB, 0, 2 * pi, 0, 2 * pi)
        EsinAsquared, _ = dblquad(fsinAsquared, 0, 2 * pi, 0, 2 * pi)
        EsinBsquared, _ = dblquad(fsinBsquared, 0, 2 * pi, 0, 2 * pi)

        rhoc = EsinAsinB / sqrt(EsinAsquared * EsinBsquared)
        return rhoc

    def mean_4D(self):
        """
        Calculates the 4D mean of [cos(x1), sin(x1), cos(x2), sin(x2)]

        Returns:
        mu (4 x 1)
            expectation value of [cos(x1), sin(x1), cos(x2), sin(x2)]
        """
        m = self.trigonometric_moment(1)
        mu = array([m[0].real, m[0].imag, m[1].real, m[1].imag]).ravel()
        return mu
