""" Abstract base class for all for all condintional densities on all domains """
from abc import abstractmethod

from .abstract_distribution import AbstractDistribution


class AbstractNonConditionalDistribution(AbstractDistribution):
    def __init__(self, dim):
        AbstractDistribution.__init__(self, dim=dim)

    @abstractmethod
    def get_manifold_size(self):
        pass
