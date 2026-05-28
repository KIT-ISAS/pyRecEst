from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .custom_hypertoroidal_distribution import CustomHypertoroidalDistribution


class CustomToroidalDistribution(
    CustomHypertoroidalDistribution, AbstractToroidalDistribution
):
    def __init__(self, f, scale_by=1, shift_by=None):
        AbstractToroidalDistribution.__init__(self)
        CustomHypertoroidalDistribution.__init__(
            self, f, 2, shift_by=shift_by, scale_by=scale_by
        )
