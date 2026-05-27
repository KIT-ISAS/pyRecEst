# pylint: disable=no-name-in-module,no-member
from beartype import beartype
from pyrecest.backend import concatenate, hstack, prod, stack

from .abstract_cart_prod_distribution import AbstractCartProdDistribution


class CartProdStackedDistribution(AbstractCartProdDistribution):
    def __init__(self, dists):
        self.dists = dists
        AbstractCartProdDistribution.__init__(self, sum(dist.dim for dist in dists))

    @beartype
    def sample(self, n: int):
        assert n > 0, "n must be positive"
        return hstack([dist.sample(n) for dist in self.dists])

    def pdf(self, xs):
        ps = []
        next_dim = 0
        for dist in self.dists:
            next_input_dim = next_dim + dist.input_dim
            if xs.ndim == 1:
                xs_curr = xs[next_dim:next_input_dim]
            else:
                xs_curr = xs[:, next_dim:next_input_dim]
            ps.append(dist.pdf(xs_curr))
            next_dim = next_input_dim
        return prod(stack(ps), axis=0)

    def shift(self, shift_by):
        assert len(shift_by) == self.dim, "Incorrect number of offsets"
        shifted_dists = []
        curr_dim = 0
        for dist in self.dists:
            shifted_dists.append(
                dist.shift(shift_by[curr_dim : curr_dim + dist.dim])  # noqa: E203
            )
            curr_dim += dist.dim
        return self.__class__(shifted_dists)

    def set_mode(self, new_mode):
        new_dists = []
        curr_ind = 0
        for dist in self.dists:
            new_dists.append(
                dist.set_mode(new_mode[curr_ind : curr_ind + dist.dim])  # noqa: E203
            )
            curr_ind += dist.dim
        return self.__class__(new_dists)

    def hybrid_mean(self):
        return concatenate([dist.mean() for dist in self.dists])

    def mean(self):
        """
        Convenient access to hybrid_mean() to have a consistent interface
        throughout manifolds.

        :return: The mean of the distribution.
        :rtype:
        """
        return self.hybrid_mean()

    def mode(self):
        return concatenate([dist.mode() for dist in self.dists])
