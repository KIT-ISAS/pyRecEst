import numpy as np

from .abstract_cart_prod_distribution import AbstractCartProdDistribution


class CartProdStackedDistribution(AbstractCartProdDistribution):
    def __init__(self, dists):
        self.dists = dists
        AbstractCartProdDistribution.__init__(self, sum(dist.dim for dist in dists))

    def sample(self, n):
        assert n > 0 and isinstance(n, int), "n must be a positive integer"
        return np.hstack([dist.sample(n) for dist in self.dists])

    def pdf(self, xs):
        ps = np.empty((len(self.dists), xs.shape[1]))
        next_dim = 0
        for i, dist in enumerate(self.dists):
            ps[i, :] = dist.pdf(xs[next_dim : next_dim + dist.dim, :])  # noqa: E203
            next_dim += dist.dim
        return np.prod(ps, axis=0)

    def shift(self, shift_by):
        assert len(shift_by) == self.dim, "Incorrect number of offsets"
        shifted_dists = [
            dist.shift(shift_by[curr_dim : curr_dim + dist.dim])  # noqa: E203
            for curr_dim, dist in enumerate(self.dists)
        ]
        return CartProdStackedDistribution(shifted_dists)

    def set_mode(self, new_mode):
        new_dists = []
        curr_ind = 0
        for dist in self.dists:
            new_dists.append(
                dist.set_mode(new_mode[curr_ind : curr_ind + dist.dim])  # noqa: E203
            )
            curr_ind += dist.dim
        return CartProdStackedDistribution(new_dists)

    def hybrid_mean(self):
        return np.concatenate([dist.mean() for dist in self.dists])

    def mean(self):
        """
        Convenient access to hybrid_mean() to have a consistent interface
        throughout manifolds.

        :return: The mean of the distribution.
        :rtype: np.ndarray
        """
        return self.hybrid_mean()

    def mode(self):
        return np.concatenate([dist.mode() for dist in self.dists])
