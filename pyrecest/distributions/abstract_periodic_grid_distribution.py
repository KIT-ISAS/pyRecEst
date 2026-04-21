from .abstract_grid_distribution import AbstractGridDistribution
from .abstract_periodic_distribution import AbstractPeriodicDistribution


class AbstractPeriodicGridDistribution(
    AbstractGridDistribution, AbstractPeriodicDistribution
):
    """Abstract base class for grid distributions on periodic domains."""
