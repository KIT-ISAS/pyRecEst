from abc import abstractmethod

import numpy as np
from scipy.integrate import dblquad, nquad, quad
from scipy.optimize import minimize

from .abstract_non_conditional_distribution import AbstractNonConditionalDistribution


class AbstractLinearDistribution(AbstractNonConditionalDistribution):
    @abstractmethod
    def pdf(self, x):
        pass

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def covariance(self):
        pass

    def get_manifold_size(self):
        return np.inf

    def mode_numerical(self):
        def neg_pdf(self, x):
            return -self.pdf(x)

        result = minimize(neg_pdf, np.zeros(self.dim), method="L-BFGS-B")
        return result.x

    def mean_numerical(self):
        if self.dim == 1:
            mu = quad(lambda x: x * self.pdf(x), -np.inf, np.inf)[0]
        elif self.dim == 2:
            mu = np.array([np.NaN, np.NaN])
            mu[0] = dblquad(
                lambda x, y: x * self.pdf(np.array([x, y])),
                -np.inf,
                np.inf,
                lambda _: -np.inf,
                lambda _: np.inf,
            )[0]
            mu[1] = dblquad(
                lambda x, y: y * self.pdf(np.array([x, y])),
                -np.inf,
                np.inf,
                lambda _: -np.inf,
                lambda _: np.inf,
            )[0]
        elif self.dim == 3:
            mu = np.array([np.NaN, np.NaN, np.NaN])

            def integrand1(x, y, z):
                return x * self.pdf(np.array([x, y, z]))

            def integrand2(x, y, z):
                return y * self.pdf(np.array([x, y, z]))

            def integrand3(x, y, z):
                return z * self.pdf(np.array([x, y, z]))

            mu[0] = nquad(
                integrand1, [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]
            )[0]
            mu[1] = nquad(
                integrand2, [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]
            )[0]
            mu[2] = nquad(
                integrand3, [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]
            )[0]
        else:
            raise ValueError(
                "Dimension currently not supported for all types of densities."
            )
        return mu

    def covariance_numerical(self):
        mu = self.mean()
        if self.dim == 1:
            C = quad(lambda x: (x - mu) ** 2 * self.pdf(x), -np.inf, np.inf)[0]
        elif self.dim == 2:
            C = np.array([[np.NaN, np.NaN], [np.NaN, np.NaN]])

            def integrand1(x, y):
                return (x - mu[0]) ** 2 * self.pdf(np.array([x, y]))

            def integrand2(x, y):
                return (x - mu[0]) * (y - mu[1]) * self.pdf(np.array([x, y]))

            def integrand3(x, y):
                return (y - mu[1]) ** 2 * self.pdf(np.array([x, y]))

            C[0, 0] = dblquad(
                integrand1, -np.inf, np.inf, lambda _: -np.inf, lambda _: np.inf
            )[0]
            C[0, 1] = dblquad(
                integrand2, -np.inf, np.inf, lambda _: -np.inf, lambda _: np.inf
            )[0]
            C[1, 1] = dblquad(
                integrand3, -np.inf, np.inf, lambda _: -np.inf, lambda _: np.inf
            )[0]
        else:
            raise NotImplementedError
