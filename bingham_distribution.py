import numpy as np
from abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
import copy

class BinghamDistribution(AbstractHypersphericalDistribution):
    def __init__(self, Z, M):
        self.dim = M.shape[0]

        assert M.shape[1] == self.dim, 'M is not square'
        assert Z.shape[0] == self.dim, 'Z has wrong number of rows'
        assert Z.ndim == 1, 'Z needs to be a 1-D vector'
        assert Z[self.dim - 1] == 0, 'Last entry of Z needs to be zero'
        assert np.all(Z[:-1] <= Z[1:]), 'Values in Z have to be ascending'

        # Enforce that M is orthogonal
        epsilon = 0.001
        assert np.max(np.abs(M @ M.T - np.eye(self.dim))) < epsilon, 'M is not orthogonal'

        self.Z = Z
        self.M = M
        self.F = self.compute_F()

    def compute_F(self):
        # Currently, only supporting numerical integration
        bd_unnormalized = copy.deepcopy(self)
        bd_unnormalized.F = 1
        return bd_unnormalized.integral_numerical()

    def compute_dF(Z):
        raise NotImplementedError('Not implemented.')

    def pdf(self, xa):
        assert xa.shape[0] == self.dim

        C = self.M @ np.diag(self.Z) @ self.M.T
        p = 1 / self.F * np.exp(np.sum(xa * (C @ xa), axis=0))
        return p

    def mean_direction(self):
        raise NotImplementedError('Due to its symmetry, the mean direction is undefined for Bingham distributions.')

    def multiply(self, B2):
        assert isinstance(B2, BinghamDistribution)
        if self.dim == B2.dim:
            C = self.M @ np.diag(self.Z.ravel()) @ self.M.T + B2.M @ np.diag(B2.Z.ravel()) @ B2.M.T  # New exponent

            C = 0.5 * (C + C.T) # Symmetrize 
            D, V = np.linalg.eig(C)
            order = np.argsort(D)  # Sort eigenvalues
            V = V[:, order]
            Z_ = D[order]
            Z_ = Z_ - Z_[-1]  # Ensure last entry is zero
            M_ = V
            return BinghamDistribution(Z_, M_)
        else:
            raise ValueError('Dimensions do not match')

    def sample(self, n):
        return self.sample_kent(n)

    def sample_kent(self, n):
        raise NotImplementedError('Not yet implemented.')