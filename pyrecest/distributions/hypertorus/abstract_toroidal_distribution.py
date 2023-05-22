import numpy as np
from scipy.integrate import dblquad

from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class AbstractToroidalDistribution(AbstractHypertoroidalDistribution):
    def __init__(self):
        AbstractHypertoroidalDistribution.__init__(self, 2)

    def covariance_4D_numerical(self):
        m = self.mean_4D()

        def f(i, j, x, y):
            funcs = [
                lambda x, _: np.cos(x) - m[0],
                lambda x, _: np.sin(x) - m[1],
                lambda _, y: np.cos(y) - m[2],
                lambda _, y: np.sin(y) - m[3],
            ]
            return self.pdf(np.array([x, y])) * funcs[i](x, y) * funcs[j](x, y)

        C = np.zeros((4, 4))
        for i in range(4):
            for j in range(i, 4):
                C[i, j], _ = dblquad(f, 0, 2 * np.pi, 0, 2 * np.pi, args=(i, j))
                if i != j:
                    C[j, i] = C[i, j]

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
