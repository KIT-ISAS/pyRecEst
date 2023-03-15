from hypertoroidal_wd_distribution import HypertoroidalWDDistribution
import numpy as np

class WDDistribution(HypertoroidalWDDistribution):
    def __init__(self, d, w=None):
        if w is None:
            w = np.ones((1, d.shape[1])) / d.shape[1]

        super().__init__(d, w)
        assert self.dim == 1, "The dimension must be 1 for WDDistribution"

    def plot(self):
        super().plot()

    def plot_interpolated(self, plot_string='-'):
        raise NotImplementedError("No interpolation available for WDDistribution.")

    def trigonometric_moment(self, n, whole_range=False, transformation='identity'):
        return super().trigonometric_moment(n)
