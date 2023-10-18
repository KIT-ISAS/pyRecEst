from pyrecest.backend import linalg
from math import pi
from pyrecest.backend import random
from typing import Union
from pyrecest.backend import vstack
from pyrecest.backend import ones
from pyrecest.backend import meshgrid
from pyrecest.backend import linspace
from pyrecest.backend import concatenate
from pyrecest.backend import int64
from pyrecest.backend import int32
from pyrecest.backend import zeros
from pyrecest.backend import array
import warnings
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from scipy.optimize import minimize

from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)


class AbstractHyperhemisphericalDistribution(AbstractHypersphereSubsetDistribution):
    def mean(self) -> np.ndarray:
        """
        Convenient access to axis to have a consistent interface
        throughout manifolds.

        :return: The mean of the distribution.
        :rtype: np.ndarray
        """
        return self.mean_axis()

    # jscpd:ignore-start
    def sample_metropolis_hastings(
        self,
        n: Union[int, int32, int64],
        burn_in: Union[int, int32, int64] = 10,
        skipping: Union[int, int32, int64] = 5,
        proposal: Callable | None = None,
        start_point: np.ndarray | None = None,
    ) -> np.ndarray:
        # jscpd:ignore-end
        if proposal is None:
            # For unimodal densities, other proposals may be far better.
            from .hyperhemispherical_uniform_distribution import (
                HyperhemisphericalUniformDistribution,
            )

            def proposal(_):
                return HyperhemisphericalUniformDistribution(self.dim).sample(1)

        if start_point is None:
            start_point = HyperhemisphericalUniformDistribution(self.dim).sample(1)

        return super().sample_metropolis_hastings(
            n, burn_in, skipping, proposal=proposal, start_point=start_point
        )

    def mean_direction_numerical(self) -> np.ndarray:
        warning_msg = (
            "The result is the mean direction on the upper hemisphere along the last dimension. "
            "It is not a mean of a symmetric distribution, which would not have a proper mean. "
            "It is also not one of the modes of the symmetric distribution since it is biased "
            "toward [0;...;0;1] because the lower half is considered inexistent."
        )
        warnings.warn(warning_msg)

        if self.dim == 1:
            mu = super().mean_direction_numerical([0, pi])
        elif self.dim <= 3:
            mu = super().mean_direction_numerical(
                [
                    zeros(self.dim),
                    [2 * pi, *pi * ones(self.dim - 2), pi / 2],
                ]
            )
        else:
            from .hyperhemispherical_uniform_distribution import (
                HyperhemisphericalUniformDistribution,
            )

            Sd = self.get_manifold_size()
            n = 10000
            r = HyperhemisphericalUniformDistribution(self.dim).sample(n)
            p = self.pdf(r)
            mu = r @ p / n * Sd

        if linalg.norm(mu) < 1e-9:
            warnings.warn(
                "Density may not have actually have a mean direction because integral yields a point very close to the origin."
            )

        mu = mu / linalg.norm(mu)
        return mu

    @staticmethod
    def get_full_integration_boundaries(dim: Union[int, int32, int64]) -> np.ndarray:
        if dim == 1:
            integration_boundaries = [0, pi]
        else:
            integration_boundaries = vstack(
                (
                    zeros(dim),
                    concatenate(
                        (array([2 * pi]), pi * ones(dim - 2), array([pi / 2]))
                    ),
                )
            ).T
        return integration_boundaries

    def integrate(self, integration_boundaries: np.ndarray | None = None) -> float:
        if integration_boundaries is None:
            integration_boundaries = (
                AbstractHyperhemisphericalDistribution.get_full_integration_boundaries(
                    self.dim
                )
            )
        return super().integrate(integration_boundaries)

    def integrate_numerically(
        self, integration_boundaries: np.ndarray | None = None
    ) -> float:
        if integration_boundaries is None:
            integration_boundaries = (
                AbstractHyperhemisphericalDistribution.get_full_integration_boundaries(
                    self.dim
                )
            )
        return super().integrate_numerically(integration_boundaries)

    @staticmethod
    def integrate_fun_over_domain(
        f_hypersph_coords: Callable, dim: Union[int, int32, int64]
    ) -> float:
        integration_boundaries = (
            AbstractHyperhemisphericalDistribution.get_full_integration_boundaries(dim)
        )
        return AbstractHypersphereSubsetDistribution.integrate_fun_over_domain_part(
            f_hypersph_coords, dim, integration_boundaries
        )

    def mode_numerical(self) -> np.ndarray:
        def objective_function_2d(s):
            return -self.pdf(AbstractHypersphereSubsetDistribution.polar_to_cart(s))

        assert self.dim == 2, "Currently only implemented for 2D hemispheres."

        s0 = random.rand(self.dim) * pi
        result = minimize(
            objective_function_2d,
            s0,
            options={
                "disp": "notify-detailed",
                "gtol": 1e-12,
                "maxiter": 2000,
            },
        )
        m = AbstractHypersphereSubsetDistribution.polar_to_cart(result.x)
        return (1 - 2 * (m[-1] < 0)) * m

    @staticmethod
    def plot_hemisphere(resolution: Union[int, int32, int64] = 150):
        x, y, z = meshgrid(
            linspace(-1, 1, resolution),
            linspace(-1, 1, resolution),
            linspace(0, 1, resolution // 2),
        )
        mask = (x**2 + y**2 + z**2 <= 1) & (z >= 0)
        x, y, z = x[mask].reshape(-1, 1), y[mask].reshape(-1, 1), z[mask].reshape(-1, 1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c="r", marker="o")
        plt.show()

    def get_manifold_size(self):
        return (
            0.5
            * AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
                self.dim
            )
        )