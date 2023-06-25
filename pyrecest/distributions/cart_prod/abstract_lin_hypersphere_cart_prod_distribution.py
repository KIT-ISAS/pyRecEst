from .abstract_lin_hypersphere_subset_cart_prod_distribution import (
    AbstractLinHypersphereSubsetCartProdDistribution,
)
from .abstract_lin_periodic_cart_prod_distribution import (
    AbstractLinPeriodicCartProdDistribution,
)


class AbstractLinHypersphereCartProdDistribution(
    AbstractLinHypersphereSubsetCartProdDistribution,
    AbstractLinPeriodicCartProdDistribution,
):
    pass
