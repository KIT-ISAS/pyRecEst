from abc import abstractmethod

from .abstract_bounded_domain_distribution import AbstractBoundedDomainDistribution


class AbstractPeriodicDistribution(AbstractBoundedDomainDistribution):
    def __init__(self, dim):
        super().__init__(dim=dim)

    def mean(self):
        return self.mean_direction()

    @abstractmethod
    def mean_direction(self):
        pass
