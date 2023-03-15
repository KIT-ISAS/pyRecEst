from abc import ABC, abstractmethod
from abstract_distribution import AbstractDistribution

class AbstractPeriodicDistribution(AbstractDistribution):
    def mean(self):
        return self.mean_direction()

    @abstractmethod
    def mean_direction(self):
        pass
