from .hypertoroidal_wd_distribution import HypertoroidalWDDistribution


class WDDistribution(HypertoroidalWDDistribution):
    def __init__(self, d, w=None):
        HypertoroidalWDDistribution.__init__(self, d, w)
        self.dim = (
            1  # Manually setting dimension because it may not be detected correctly
        )
        assert self.d.shape[0] == self.w.shape[0]

    def plot(self):
        super().plot()

    def plot_interpolated(self, plot_string="-"):
        raise NotImplementedError("No interpolation available for WDDistribution.")

    def trigonometric_moment(self, n, whole_range=False, transformation="identity"):
        return super().trigonometric_moment(n)
