from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)


class ToroidalWrappedNormalDistribution(
    HypertoroidalWrappedNormalDistribution, AbstractToroidalDistribution
):
    pass
