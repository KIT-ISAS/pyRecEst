import numpy as np
from scipy.stats import multivariate_normal
from itertools import product
from AbstractHypertoroidalDistribution import AbstractHypertoroidalDistribution

class HypertoroidalWNDistribution(AbstractHypertoroidalDistribution):
    def __init__(self, mu_, C_):
        assert C_.shape[0] == C_.shape[1], "C must be dim x dim"
        assert np.allclose(C_, C_.T, atol=1e-8), "C must be symmetric"
        assert np.all(np.linalg.eigvals(C_) > 0), "C must be positive definite"
        assert mu_.shape[0] == C_.shape[1], "mu must be dim x 1"

        self.mu = np.mod(mu_, 2 * np.pi)
        self.C = C_
        self.dim = mu_.shape[0]

    def pdf(self, xa, m=3):
        xa = np.asarray(xa)
        dim = self.mu.shape[0]

        # Generate all combinations of offsets for each dimension
        offsets = [np.arange(-m, m + 1) * 2 * np.pi for _ in range(dim)]
        offset_combinations = np.array(np.meshgrid(*offsets)).T.reshape(-1, dim)

        # Calculate the PDF values by considering all combinations of offsets
        pdf_values = np.zeros(xa.shape[1])
        for offset in offset_combinations:
            shifted_xa = xa + offset[:, np.newaxis]
            pdf_values += multivariate_normal.pdf(shifted_xa.T, mean=self.mu.flatten(), cov=self.C)

        return pdf_values
