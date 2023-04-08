""" Abstract base class for all filters for Euclidean domains """
from abc import abstractmethod
from .abstract_filter import AbstractFilter

class AbstractEuclideanFilter(AbstractFilter):
    """ Abstract base class for all filters for Euclidean domains """
    @abstractmethod
    def get_estimate(self):
        pass

    def get_point_estimate(self):
        est = self.get_estimate().mean()
        return est

    def get_estimate_mean(self):
        mean = self.get_point_estimate()
        return mean
