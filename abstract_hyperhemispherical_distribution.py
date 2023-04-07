import warnings
import numpy as np
from scipy.optimize import minimize
from abstract_hypersphere_subset_distribution import AbstractHypersphereSubsetDistribution
from abstract_hyperspherical_distribution import AbstractHypersphericalDistribution

class AbstractHyperhemisphericalDistribution(AbstractHypersphereSubsetDistribution):
    def mean(self):
        return self.mean_axis()

    def mean_direction_numerical(self):
        warning_msg = ('The result is the mean direction on the upper hemisphere along the last dimension. '
                       'It is not a mean of a symmetric distribution, which would not have a proper mean. '
                       'It is also not one of the modes of the symmetric distribution since it is biased '
                       'toward [0;...;0;1] because the lower half is considered inexistent.')
        warnings.warn(warning_msg)
        
        if self.dim == 1:
            mu = super().mean_direction_numerical([0, np.pi])
        elif self.dim <= 3:
            mu = super().mean_direction_numerical([np.zeros(self.dim), [2 * np.pi, *np.pi * np.ones(self.dim - 2), np.pi / 2]])
        else:
            Sd = self.get_manifold_size()
            n = 10000
            r = HyperhemisphericalUniformDistribution(self.dim).sample(n)
            p = self.pdf(r)
            mu = r @ p / n * Sd

        if np.linalg.norm(mu) < 1e-9:
            warnings.warn('Density may not have actually have a mean direction because integral yields a point very close to the origin.')

        mu = mu / np.linalg.norm(mu)
        return mu

    def moment_numerical(self):
        if self.dim == 1:
            return super().moment_numerical([0, np.pi])
        else:
            return super().moment_numerical([np.zeros(self.dim - 1), [2 * np.pi, *np.pi * np.ones(self.dim - 3), np.pi / 2]])

    def integral_numerical(self):
        if self.dim == 1:
            return super().integral_numerical([0, np.pi])
        elif self.dim <= 3:
            return super().integral_numerical(np.vstack((np.zeros(self.dim), np.hstack((2 * np.pi, np.pi * np.ones(self.dim - 2), np.pi / 2)))).T)
        else:
            from hyperhemispherical_uniform_distribution import HyperhemisphericalUniformDistribution
            n = 10000
            r = HyperhemisphericalUniformDistribution(self.dim).sample(n)
            p = self.pdf(r)
            Sd = AbstractHypersphericalDistribution.compute_unit_sphere_surface(self.dim)
            i = np.sum(p) / n * Sd 
            return i

    def mode_numerical(self):
        def objective_function(s):
            return -self.pdf(polar2cart(s))

        s0 = np.random.rand(self.dim) * np.pi
        result = minimize(objective_function, s0, options={'disp': 'notify-detailed', 'gtol': 1e-12, 'maxiter': 2000, 'xtol': 1e-12})
        m = polar2cart(result.x)
        return m

    def sample_metropolis_hastings(self, n, proposal=None, start_point=None, burn_in=10, skipping=5):
        if proposal is None:
            def normalize(x):
                return x / np.linalg.norm(x)

            def to_upper_hemisphere(s):
                return (1 - 2 * (s[-1] < 0)) * s

            proposal = lambda x: to_upper_hemisphere(normalize(x + np.random.normal(0, 1, (self.dim, 1))))

        if start_point is None:
            start_point = self.mode()

        s = super().sample_metropolis_hastings(n, proposal, start_point, burn_in, skipping)
        return s

    def get_manifold_size(self):
        return 0.5 * AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(self.dim)