from abc import abstractmethod

from .abstract_bounded_distribution import AbstractBoundedDistribution


class AbstractPeriodicDistribution(AbstractBoundedDistribution):
    def __init__(self, dim):
        super().__init__(dim=dim)

    def mean(self):
        return self.mean_direction()

    @abstractmethod
    def mean_direction(self):
        pass
