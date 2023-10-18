from pyrecest.backend import diag
from pyrecest.backend import linalg
from math import pi
from pyrecest.backend import sum
from pyrecest.backend import eye
from pyrecest.backend import exp
from pyrecest.backend import all
from pyrecest.backend import abs
import numpy as np
from scipy.integrate import quad
from scipy.special import iv

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class BinghamDistribution(AbstractHypersphericalDistribution):
    def __init__(self, Z: np.ndarray, M: np.ndarray):
        AbstractHypersphericalDistribution.__init__(self, M.shape[0] - 1)

        assert M.shape[1] == self.input_dim, "M is not square"
        assert Z.shape[0] == self.input_dim, "Z has wrong length"
        assert Z.ndim == 1, "Z needs to be a 1-D vector"
        assert Z[-1] == 0, "Last entry of Z needs to be zero"
        assert all(Z[:-1] <= Z[1:]), "Values in Z have to be ascending"

        # Verify that M is orthogonal
        epsilon = 0.001
        assert (
            np.max(abs(M @ M.T - eye(self.dim + 1))) < epsilon
        ), "M is not orthogonal"

        self.Z = Z
        self.M = M
        self._F = None
        self._dF = None

    @property
    def F(self):
        if self._F is None:
            # Currently, only supporting numerical integration
            # Temporarily set _F to 1 to use .integrate_numerically to calculate the normalization constant
            self._F = 1
            self._F = self.integrate_numerically()
        return self._F

    @F.setter
    def F(self, value):
        self._F = value

    @staticmethod
    def calculate_F(Z):
        """Uses method by wood. Only supports 4-D distributions."""
        assert Z.shape[0] == 4

        def J(Z, u):
            return iv(0, 0.5 * abs(Z[0] - Z[1]) * u) * iv(
                0, 0.5 * abs(Z[2] - Z[3]) * (1 - u)
            )

        def ifun(u):
            return J(Z, u) * exp(
                0.5 * (Z[0] + Z[1]) * u + 0.5 * (Z[2] + Z[3]) * (1 - u)
            )

        return 2 * pi**2 * quad(ifun, 0, 1)[0]

    def pdf(self, xs):
        assert xs.shape[-1] == self.dim + 1

        C = self.M @ diag(self.Z) @ self.M.T
        p = 1 / self.F * exp(sum(xs.T * (C @ xs.T), axis=0))
        return p

    def mean_direction(self):
        raise NotImplementedError(
            "Due to its symmetry, the mean direction is undefined for Bingham distributions."
        )

    def multiply(self, B2):
        assert isinstance(B2, BinghamDistribution)
        if self.dim != B2.dim:
            raise ValueError("Dimensions do not match")

        C = (
            self.M @ diag(self.Z.ravel()) @ self.M.T
            + B2.M @ diag(B2.Z.ravel()) @ B2.M.T
        )  # New exponent

        C = 0.5 * (C + C.T)  # Symmetrize
        D, V = linalg.eig(C)
        order = np.argsort(D)  # Sort eigenvalues
        V = V[:, order]
        Z_ = D[order]
        Z_ = Z_ - Z_[-1]  # Ensure last entry is zero
        M_ = V
        return BinghamDistribution(Z_, M_)

    def sample(self, n):
        return self.sample_metropolis_hastings(n)

    @property
    def dF(self):
        if self._dF is None:
            self._dF = self.calculate_dF()
        return self._dF

    def calculate_dF(self):
        dim = self.Z.shape[0]  # Assuming Z is a property of the object
        dF = zeros(dim)
        epsilon = 0.001
        for i in range(dim):
            # Using finite differences
            dZ = zeros(dim)
            dZ[i] = epsilon
            F1 = self.calculate_F(self.Z + dZ)
            F2 = self.calculate_F(self.Z - dZ)
            dF[i] = (F1 - F2) / (2 * epsilon)
        return dF

    def sample_kent(self, n):
        raise NotImplementedError("Not yet implemented.")

    def moment(self):
        """
        Returns:
            S (numpy.ndarray): scatter/covariance matrix in R^d
        """
        D = diag(self.dF / self.F)
        # It should already be normalized, but numerical inaccuracies can lead to values unequal to 1
        D = D / sum(diag(D))
        S = self.M @ D @ self.M.T
        S = (S + S.T) / 2  # Enforce symmetry
        return S