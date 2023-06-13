import numpy as np

from .abstract_lin_bounded_cart_prod_distribution import (
    AbstractLinBoundedCartProdDistribution,
)


class AbstractLinPeriodicCartProdDistribution(AbstractLinBoundedCartProdDistribution):
    """
    For Cartesian products of linear and periodic domains. Assumption is
    that it is bounded x R^n (in this order)
    """

    def __init__(self, bound_dim, lin_dim, periodic_manifold_type="unspecified"):
        AbstractLinBoundedCartProdDistribution.__init__(self, bound_dim, lin_dim)
        self._allowed_periodic_manifold_types = [
            "unspecified",
            "hypertorus",
            "hypersphere",
            "hyperhemisphere",
        ]
        self.periodic_manifold_type = periodic_manifold_type

    def get_manifold_size(self):
        assert (
            self.lin_dim > 0
        ), "This class is not intended to be used for purely periodic domains."
        return np.inf

    @property
    def input_dim(self):
        if self.periodic_manifold_type == "unspecified":
            raise ValueError(
                "Cannot provide input_dim for unspecified pariodic domain type."
            )

        if self.periodic_manifold_type in ["hypersphere", "hyperhemisphere"]:
            return self.bound_dim + 1 + self.lin_dim

        # Only hypertorus remains
        return self.bound_dim + self.lin_dim

    @property
    def periodic_manifold_type(self):
        return self._periodic_manifold_type

    @periodic_manifold_type.setter
    def periodic_manifold_type(self, value):
        if value not in self._allowed_periodic_manifold_types:
            raise ValueError(
                "Invalid periodic manifold type. Allowed types are: "
                + str(self._allowed_periodic_manifold_types)
            )
        self._periodic_manifold_type = value
