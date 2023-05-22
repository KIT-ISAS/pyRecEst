""" Abstract base class for all filters for Euclidean domains """
from .abstract_filter import AbstractFilter


class AbstractEuclideanFilter(AbstractFilter):
    """Abstract base class for all filters for Euclidean domains"""

    def get_point_estimate(self):
        est = self.get_estimate().mean()
        return est
