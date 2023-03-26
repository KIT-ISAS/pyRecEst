from abstract_filter import AbstractFilter
from abc import abstractmethod

class AbstractEuclideanFilter(AbstractFilter):
    @abstractmethod
    def get_estimate(self):
        pass

    def get_point_estimate(self):
        est = self.get_estimate().mean()
        return est

    def get_estimate_mean(self):
        mean = self.get_point_estimate()
        return mean
