import warnings

import numpy as np
from scipy.optimize import minimize

from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)


class AbstractHyperhemisphericalDistribution(AbstractHypersphereSubsetDistribution):
    def mean(self):
        return self.mean_axis()

    def mean_direction_numerical(self):
        warning_msg = (
            "The result is the mean direction on the upper hemisphere along the last dimension. "
            "It is not a mean of a symmetric distribution, which would not have a proper mean. "
            "It is also not one of the modes of the symmetric distribution since it is biased "
            "toward [0;...;0;1] because the lower half is considered inexistent."
        )
        warnings.warn(warning_msg)

        if self.dim == 1:
            mu = super().mean_direction_numerical([0, np.pi])
        elif self.dim <= 3:
            mu = super().mean_direction_numerical(
                [
                    np.zeros(self.dim),
                    [2 * np.pi, *np.pi * np.ones(self.dim - 2), np.pi / 2],
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

        if np.linalg.norm(mu) < 1e-9:
            warnings.warn(
                "Density may not have actually have a mean direction because integral yields a point very close to the origin."
            )

        mu = mu / np.linalg.norm(mu)
        return mu

    def moment_numerical(self):
        if self.dim == 1:
            return super().moment_numerical([0, np.pi])
        else:
            return super().moment_numerical(
                [
                    np.zeros(self.dim - 1),
                    [2 * np.pi, *np.pi * np.ones(self.dim - 3), np.pi / 2],
                ]
            )

    def _get_full_integration_boundaries(self):
        if self.dim == 1:
            integration_boundaries = [0, np.pi]
        else:
            integration_boundaries = np.vstack(
                (
                    np.zeros((self.dim)),
                    np.concatenate(
                        ([2 * np.pi], np.pi * np.ones(self.dim - 2), [np.pi / 2])
                    ),
                )
            ).T
        return integration_boundaries

    def integral(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self._get_full_integration_boundaries()
        return super().integral(integration_boundaries)

    def integral_numerical(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self._get_full_integration_boundaries()
        return super().integral_numerical(integration_boundaries)

    def mode_numerical(self):
        def objective_function(s):
            return -self.pdf(AbstractHypersphereSubsetDistribution.polar2cart(s))

        s0 = np.random.rand(self.dim) * np.pi
        result = minimize(
            objective_function,
            s0,
            options={
                "disp": "notify-detailed",
                "gtol": 1e-12,
                "maxiter": 2000,
                "xtol": 1e-12,
            },
        )
        m = AbstractHypersphereSubsetDistribution.polar2cart(result.x)
        return (1 - 2 * (m[-1] < 0)) * m

    def sample_metropolis_hastings(
        self, n, proposal=None, start_point=None, burn_in=10, skipping=5
    ):
        if proposal is None:

            def normalize(x):
                return x / np.linalg.norm(x)

            def to_upper_hemisphere(s):
                return (1 - 2 * (s[-1] < 0)) * s

            def proposal(x):
                return to_upper_hemisphere(
                    normalize(x + np.random.normal(0, 1, (self.dim, 1)))
                )

        if start_point is None:
            start_point = self.mode()

        s = super().sample_metropolis_hastings(
            n, proposal, start_point, burn_in, skipping
        )
        return s

    def get_manifold_size(self):
        return (
            0.5
            * AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
                self.dim
            )
        )
