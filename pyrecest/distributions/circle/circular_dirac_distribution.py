from ..hypertorus.hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution


class CircularDiracDistribution(HypertoroidalDiracDistribution):
    def __init__(self, d, w=None):
        HypertoroidalDiracDistribution.__init__(self, d, w)
        self.dim = (
            1  # Manually setting dimension because it may not be detected correctly
        )
        assert self.d.shape[0] == self.w.shape[0]

    def plot(self):
        super().plot()

    def plot_interpolated(self, plot_string="-"):
        raise NotImplementedError("No interpolation available for WDDistribution.")

