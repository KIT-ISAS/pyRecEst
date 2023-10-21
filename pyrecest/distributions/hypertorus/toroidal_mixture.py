

from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .hypertoroidal_mixture import HypertoroidalMixture


class ToroidalMixture(HypertoroidalMixture, AbstractToroidalDistribution):
    def __init__(self, hds: list[AbstractToroidalDistribution], w):
        """
        Constructor

        :param hds: list of toroidal distributions
        :param w: list of weights
        """
        assert all(
            isinstance(hd, AbstractToroidalDistribution) for hd in hds
        ), "hds must be a list of toroidal distributions"

        HypertoroidalMixture.__init__(self, hds, w)
        AbstractToroidalDistribution.__init__(self)