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

            return self.integrate_numerically(starting_point_mod, x_mod)

        return np.array([cdf_single(x) for x in xs])

    def to_vm(self):
        """
        Convert to von Mises by trigonometric moment matching.

        Returns:
            vm (VMDistribution): Distribution with the same first trigonometric moment.
        """
        from .von_mises_distribution import VonMisesDistribution

        vm = VonMisesDistribution.from_moment(self.trigonometric_moment(1))
        return vm

    def to_wn(self):
        """
        Convert to wrapped normal by trigonometric moment matching.

        Returns:
            wn (WrappedNormalDistribution): Distribution with the same first trigonometric moment.
        """
        from .wrapped_normal_distribution import WrappedNormalDistribution

        wn = WrappedNormalDistribution.from_moment(self.trigonometric_moment(1))
        return wn

    @staticmethod
    def plot_circle(*args, **kwargs):
        theta = np.append(np.linspace(0, 2 * np.pi, 320), 0)
        p = plt.plot(np.cos(theta), np.sin(theta), *args, **kwargs)
        return p
