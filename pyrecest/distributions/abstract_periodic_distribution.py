from abc import abstractmethod

from .abstract_non_conditional_distribution import AbstractNonConditionalDistribution


class AbstractPeriodicDistribution(AbstractNonConditionalDistribution):
    def __init__(self, dim):
        super().__init__(dim=dim)

    def mean(self):
        return self.mean_direction()

    @abstractmethod
    def mean_direction(self):
        pass
