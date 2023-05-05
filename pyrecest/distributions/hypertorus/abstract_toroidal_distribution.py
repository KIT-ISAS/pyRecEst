from abc import abstractmethod

import numpy as np
from scipy.integrate import dblquad

from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class AbstractToroidalDistribution(AbstractHypertoroidalDistribution):
    def __init__(self):
        super().__init__(2)

    @abstractmethod
    def pdf(self, xs):
        pass

    def integrate(self, left=None, right=None):
        left, right = self.prepare_integral_arguments(left, right)
        return self.integrate_numerically(left, right)

    def prepare_integral_arguments(self, left=None, right=None):
        if left is None:
            left = np.array([0, 0])

        if right is None:
            right = np.array([2 * np.pi, 2 * np.pi])

        assert left.shape == (self.dim,)
        assert right.shape == (self.dim,)

        return left, right

    def covariance_4D_numerical(self):
        m = self.mean_4D()

        def f1(x, y):
            return np.cos(x) - m[0]

        def f2(x, y):
            return np.sin(x) - m[1]

        def f3(x, y):
            return np.cos(y) - m[2]

        def f4(x, y):
            return np.sin(y) - m[3]

        def f11(x, y):
            return self.pdf(np.array([x, y])) * f1(x, y) ** 2

        def f12(x, y):
            return self.pdf(np.array([x, y])) * f1(x, y) * f2(x, y)

        def f13(x, y):
            return self.pdf(np.array([x, y])) * f1(x, y) * f3(x, y)

        def f14(x, y):
            return self.pdf(np.array([x, y])) * f1(x, y) * f4(x, y)

        def f22(x, y):
            return self.pdf(np.array([x, y])) * f2(x, y) ** 2

        def f23(x, y):
            return self.pdf(np.array([x, y])) * f2(x, y) * f3(x, y)

        def f24(x, y):
            return self.pdf(np.array([x, y])) * f2(x, y) * f4(x, y)

        def f33(x, y):
            return self.pdf(np.array([x, y])) * f3(x, y) ** 2

        def f34(x, y):
            return self.pdf(np.array([x, y])) * f3(x, y) * f4(x, y)

        def f44(x, y):
            return self.pdf(np.array([x, y])) * f4(x, y) ** 2

        c11, _ = dblquad(f11, 0, 2 * np.pi, 0, 2 * np.pi)
        c12, _ = dblquad(f12, 0, 2 * np.pi, 0, 2 * np.pi)
        c13, _ = dblquad(f13, 0, 2 * np.pi, 0, 2 * np.pi)
        c14, _ = dblquad(f14, 0, 2 * np.pi, 0, 2 * np.pi)
        c22, _ = dblquad(f22, 0, 2 * np.pi, 0, 2 * np.pi)
        c23, _ = dblquad(f23, 0, 2 * np.pi, 0, 2 * np.pi)
        c24, _ = dblquad(f24, 0, 2 * np.pi, 0, 2 * np.pi)
        c33, _ = dblquad(f33, 0, 2 * np.pi, 0, 2 * np.pi)
        c34, _ = dblquad(f34, 0, 2 * np.pi, 0, 2 * np.pi)
        c44, _ = dblquad(f44, 0, 2 * np.pi, 0, 2 * np.pi)

        C = np.array(
            [
                [c11, c12, c13, c14],
                [c12, c22, c23, c24],
                [c13, c23, c33, c34],
                [c14, c24, c34, c44],
            ]
        )
        return C

    def circular_correlation_jammalamadaka(self):
        rhoc = self.circular_correlation_jammalamadaka_numerical()
        return rhoc

    def circular_correlation_jammalamadaka_numerical(self):
        m = self.mean_direction()

        def fsinAsinB(x, y):
            return self.pdf(np.array([x, y])) * np.sin(x - m[0]) * np.sin(y - m[1])

        def fsinAsquared(x, y):
            return self.pdf(np.array([x, y])) * np.sin(x - m[0]) ** 2

        def fsinBsquared(x, y):
            return self.pdf(np.array([x, y])) * np.sin(y - m[1]) ** 2

        EsinAsinB, _ = dblquad(fsinAsinB, 0, 2 * np.pi, 0, 2 * np.pi)
        EsinAsquared, _ = dblquad(fsinAsquared, 0, 2 * np.pi, 0, 2 * np.pi)
        EsinBsquared, _ = dblquad(fsinBsquared, 0, 2 * np.pi, 0, 2 * np.pi)

        rhoc = EsinAsinB / np.sqrt(EsinAsquared * EsinBsquared)
        return rhoc

    def mean_4D(self):
        """
        Calculates the 4D mean of [cos(x1), sin(x1), cos(x2), sin(x2)]

        Returns:
        mu (4 x 1)
            expectation value of [cos(x1), sin(x1), cos(x2), sin(x2)]
        """
        m = self.trigonometric_moment(1)
        mu = np.array([m[0].real, m[0].imag, m[1].real, m[1].imag]).ravel()
        return mu
