from .abstract_bounded_domain_distribution import AbstractBoundedDomainDistribution


class AbstractBoundedNonPeriodicDistribution(AbstractBoundedDomainDistribution):
    def mean(self):
        raise ValueError("Mean currently not supported.")
