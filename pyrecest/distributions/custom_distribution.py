import copy

from .abstract_distribution import AbstractDistribution


class CustomDistribution(AbstractDistribution):
    def __init__(self, f, dim: int):
        self.dim = dim
        self.f = f
        self.scale_by = 1
        self.shift_by = 0

    def shift(self, shift_vector):
        assert self.dim == shift_vector.shape[0]
        cd = copy.deepcopy(self)
        cd.shift_by = self.shift_by + shift_vector
        return cd
