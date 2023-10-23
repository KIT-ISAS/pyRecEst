from .abstract_bounded_domain_distribution import AbstractBoundedDomainDistribution


class AbstractBoundedNonPeriodicDistribution(AbstractBoundedDomainDistribution):
    """
    Abstract class for distributions with on non-periodic bounded domains.

    This class extends the AbstractBoundedDomainDistribution class, and
    serves as the base class for all distributions that are
    defined over non-periodic bounded domains.
    """

    def mean(self):
        raise ValueError("Mean currently not supported.")
