from .abstract_nonperiodic_distribution import AbstractNonperiodicDistribution
from .abstract_custom_distribution import AbstractCustomDistribution


class AbstractCustomNonPeriodicDistribution(
    AbstractCustomDistribution, AbstractNonperiodicDistribution
):
    """
    This class serves as a base for all custom non-periodic distributions.

    Custom non-periodic distributions are distributions that are defined by a
    given probability density function and are not periodic.
    """
