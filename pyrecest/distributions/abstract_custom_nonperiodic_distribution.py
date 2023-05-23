from .abstract_custom_distribution import AbstractCustomDistribution
from .abstract_nonperiodic_distribution import AbstractNonperiodicDistribution


class AbstractCustomNonPeriodicDistribution(
    AbstractCustomDistribution, AbstractNonperiodicDistribution
):
    pass
