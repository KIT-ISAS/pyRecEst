from ..hypertorus.hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution
from .abstract_circular_distribution import AbstractCircularDistribution


class CircularDiracDistribution(
    HypertoroidalDiracDistribution, AbstractCircularDistribution
):
    def __init__(self, d, w=None):
        AbstractCircularDistribution.__init__(self)
        HypertoroidalDiracDistribution.__init__(self, d, w, dim=1)
        assert self.d.shape[0] == self.w.shape[0]

    def plot(self):
        super().plot()

    def plot_interpolated(self, plot_string="-"):
        raise NotImplementedError("No interpolation available for WDDistribution.")
