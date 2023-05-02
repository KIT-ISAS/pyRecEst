import matplotlib.pyplot as plt
import numpy as np

from ..hypertorus.abstract_hypertoroidal_distribution import (
    AbstractHypertoroidalDistribution,
)


class AbstractCircularDistribution(AbstractHypertoroidalDistribution):
    def __init__(self):
        AbstractHypertoroidalDistribution.__init__(self, dim=1)

    def cdf_numerical(self, xs, starting_point=0):
        xa = np.asarray(xs)
        assert xa.ndim == 1, "xa must be a 1D array"

        def cdf_single(x):
            starting_point_mod = starting_point % (2 * np.pi)
            x_mod = x % (2 * np.pi)

            if x_mod < starting_point_mod:
                return 1 - self.integrate_numerically(x_mod, starting_point_mod)
            else:
                return self.integrate_numerically(starting_point_mod, x_mod)

        return np.array([cdf_single(x) for x in xs])

    def sample_metropolis_hastings(
        self, n, proposal=None, start_point=None, burn_in=10, skipping=5
    ):
        from .wrapped_normal_distribution import WNDistribution

        if proposal is None:
            wn = WNDistribution.from_moment(self.trigonometric_moment(1))
            wn.mu = 0

            def proposal(x):
                return (x + wn.sample(1)) % (2 * np.pi)

        if start_point is None:
            start_point = self.mean_direction()

        s = super().sample_metropolis_hastings(
            n, proposal, start_point, burn_in, skipping
        )
        return s

    def to_vm(self):
        """
        Convert to von Mises by trigonometric moment matching.

        Returns:
            vm (VMDistribution): Distribution with the same first trigonometric moment.
        """
        from .von_mises_distribution import VMDistribution

        vm = VMDistribution.from_moment(self.trigonometric_moment(1))
        return vm

    def to_wn(self):
        """
        Convert to wrapped normal by trigonometric moment matching.

        Returns:
            wn (WNDistribution): Distribution with the same first trigonometric moment.
        """
        from .wrapped_normal_distribution import WNDistribution

        wn = WNDistribution.from_moment(self.trigonometric_moment(1))
        return wn

    @staticmethod
    def plot_circle(*args, **kwargs):
        theta = np.append(np.linspace(0, 2 * np.pi, 320), 0)
        p = plt.plot(np.cos(theta), np.sin(theta), *args, **kwargs)
        return p
