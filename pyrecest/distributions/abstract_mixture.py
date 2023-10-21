from pyrecest.backend import random
from typing import Union
from pyrecest.backend import sum
from pyrecest.backend import ones
from pyrecest.backend import int64
from pyrecest.backend import int32
from pyrecest.backend import empty
from pyrecest.backend import zeros
from pyrecest.backend import nonzero
import collections
import copy
import warnings


from beartype import beartype

from .abstract_distribution_type import AbstractDistributionType
from .abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)


class AbstractMixture(AbstractDistributionType):
    """
    Abstract base class for mixture distributions.
    """

    def __init__(
        self,
        dists: collections.abc.Sequence[AbstractManifoldSpecificDistribution],
        weights=None,
    ):
        AbstractDistributionType.__init__(self)
        dists = copy.deepcopy(dists)  # To prevent modifying the original object
        num_distributions = len(dists)

        if weights is None:
            weights = ones(num_distributions) / num_distributions

        if num_distributions != len(weights):
            raise ValueError("Sizes of distributions and weights must be equal")

        if not all(dists[0].dim == dist.dim for dist in dists):
            raise ValueError("All distributions must have the same dimension")

        non_zero_indices = nonzero(weights)[0]

        if len(non_zero_indices) < len(weights):
            warnings.warn(
                "Elements with zero weights detected. Pruning elements in mixture with weight zero."
            )
            dists = [dists[i] for i in non_zero_indices]
            weights = weights[non_zero_indices]

        self.dists = dists

        if abs(sum(weights) - 1.0) > 1e-10:
            warnings.warn("Weights of mixture do not sum to one.")
            self.w = weights / sum(weights)
        else:
            self.w = weights

    @property
    def input_dim(self) -> int:
        return self.dists[0].input_dim

    def sample(self, n: Union[int, int32, int64]):

        occurrences = random.multinomial(n, self.w)
        count = 0
        s = empty((n, self.input_dim))
        for i, occ in enumerate(occurrences):
            if occ != 0:
                s[count : count + occ, :] = self.dists[i].sample(occ)  # noqa: E203
                count += occ

        return s

    def pdf(self, xs):
        assert xs.shape[-1] == self.input_dim, "Dimension mismatch"

        p = zeros(1) if xs.ndim == 1 else zeros(xs.shape[0])

        for i, dist in enumerate(self.dists):
            p += self.w[i] * dist.pdf(xs)

        return p