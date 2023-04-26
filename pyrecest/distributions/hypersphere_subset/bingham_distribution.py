import numpy as np

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class BinghamDistribution(AbstractHypersphericalDistribution):
    def __init__(self, Z, M):
        self.dim = M.shape[0] - 1

        assert M.shape[1] == self.dim + 1, "M is not square"
        assert Z.shape[0] == self.dim + 1, "Z has wrong length"
        assert Z.ndim == 1, "Z needs to be a 1-D vector"
        assert Z[-1] == 0, "Last entry of Z needs to be zero"
        assert np.all(Z[:-1] <= Z[1:]), "Values in Z have to be ascending"

        # Verify that M is orthogonal
        epsilon = 0.001
        assert (
            np.max(np.abs(M @ M.T - np.eye(self.dim + 1))) < epsilon
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

    @property
    def dF(self):
        raise NotImplementedError("Not implemented.")

    def pdf(self, xs):
        assert xs.shape[-1] == self.dim + 1

        C = self.M @ np.diag(self.Z) @ self.M.T
        p = 1 / self.F * np.exp(np.sum(xs.T * (C @ xs.T), axis=0))
        return p

    def mean_direction(self):
        raise NotImplementedError(
            "Due to its symmetry, the mean direction is undefined for Bingham distributions."
        )

    def multiply(self, B2):
        assert isinstance(B2, BinghamDistribution)
        if self.dim == B2.dim:
            C = (
                self.M @ np.diag(self.Z.ravel()) @ self.M.T
                + B2.M @ np.diag(B2.Z.ravel()) @ B2.M.T
            )  # New exponent

            C = 0.5 * (C + C.T)  # Symmetrize
            D, V = np.linalg.eig(C)
            order = np.argsort(D)  # Sort eigenvalues
            V = V[:, order]
            Z_ = D[order]
            Z_ = Z_ - Z_[-1]  # Ensure last entry is zero
            M_ = V
            return BinghamDistribution(Z_, M_)
        else:
            raise ValueError("Dimensions do not match")

    def sample(self, n):
        return self.sample_kent(n)

    def sample_kent(self, n):
        raise NotImplementedError("Not yet implemented.")
