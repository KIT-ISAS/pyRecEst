import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import nquad

from ..abstract_periodic_distribution import AbstractPeriodicDistribution


class AbstractHypertoroidalDistribution(AbstractPeriodicDistribution):
    def __init__(self, dim):
        super().__init__(dim=dim)

    @staticmethod
    def integrate_fun_over_domain(f, dim):
        integration_boundaries = [(0, 2 * np.pi)] * dim
        return AbstractHypertoroidalDistribution.integrate_fun_over_domain_part(
            f, dim, integration_boundaries
        )

    @staticmethod
    def integrate_fun_over_domain_part(f, dim, integration_boundaries):
        if len(integration_boundaries) != dim:
            raise ValueError(
                "The length of integration_boundaries must match the specified dimension."
            )

        return nquad(f, integration_boundaries)[0]

    def integrate_numerically(self, integration_boundaries):
        left = integration_boundaries[0]
        right = integration_boundaries[1]
        
        def pdf_fun(*args):
            return self.pdf(np.array(args))

        integration_boundaries = [(left[i], right[i]) for i in range(self.dim)]
        return self.integrate_fun_over_domain_part(
            pdf_fun, self.dim, integration_boundaries
        )

    def trigonometric_moment_numerical(self, n):
        def moment_fun(*args):
            x = np.array(args)
            return self.pdf(x) * np.exp(1j * n * x)

        return self.integrate_fun_over_domain(moment_fun, self.dim)

    def entropy_numerical(self):
        def entropy_fun(*args):
            x = np.array(args)
            pdf_val = self.pdf(x)
            return pdf_val * np.log(pdf_val)

        return -self.integrate_fun_over_domain(entropy_fun, self.dim)

    def get_manifold_size(self):
        return (2 * np.pi) ** self.dim

    @staticmethod
    def angular_error(alpha, beta):
        assert not np.isnan(alpha).any() and not np.isnan(beta).any()

        alpha = np.mod(alpha, 2 * np.pi)
        beta = np.mod(beta, 2 * np.pi)

        diff = np.abs(alpha - beta)
        e = np.minimum(diff, 2 * np.pi - diff)

        return e

    def hellinger_distance_numerical(self, other):
        assert isinstance(other, AbstractHypertoroidalDistribution)
        assert (
            self.dim == other.dim
        ), "Cannot compare distributions with different number of dimensions."

        def hellinger_dist_fun(*args):
            x = np.array(args)
            return (np.sqrt(self.pdf(x)) - np.sqrt(other.pdf(x))) ** 2

        dist = 0.5 * self.integrate_fun_over_domain(hellinger_dist_fun, self.dim)
        return dist

    def total_variation_distance_numerical(self, other):
        assert isinstance(other, AbstractHypertoroidalDistribution)
        assert (
            self.dim == other.dim
        ), "Cannot compare distributions with different number of dimensions"

        def total_variation_dist_fun(*args):
            x = np.array(args)
            return abs(self.pdf(x) - other.pdf(x))

        dist = 0.5 * self.integrate_fun_over_domain(total_variation_dist_fun, self.dim)
        return dist

    def plot(self, *args, resolution=None):
        if resolution is None:
            resolution = 128

        if self.dim == 1:
            theta = np.linspace(0, 2 * np.pi, resolution)
            f_theta = self.pdf(theta)
            p = plt.plot(theta, f_theta, *args)
            AbstractHypertoroidalDistribution.setup_axis_circular("x")
        elif self.dim == 2:
            step = 2 * np.pi / resolution
            alpha, beta = np.meshgrid(
                np.arange(0, 2 * np.pi, step), np.arange(0, 2 * np.pi, step)
            )
            f = self.pdf(np.vstack((alpha.ravel(), beta.ravel())))
            f = f.reshape(alpha.shape)
            p = plt.contourf(alpha, beta, f, *args)
            AbstractHypertoroidalDistribution.setup_axis_circular("x")
            AbstractHypertoroidalDistribution.setup_axis_circular("y")
        elif self.dim == 3:
            raise NotImplementedError(
                "Plotting for this dimension is currently not supported"
            )
        else:
            raise ValueError("Plotting for this dimension is currently not supported")
        plt.show()
        return p

    def mean_direction(self):
        a = self.trigonometric_moment(1)
        m = np.mod(np.angle(a), 2 * np.pi)
        return m

    def mode(self):
        return self.mode_numerical()

    def mode_numerical(self):
        # Implement the optimization function fminunc equivalent in Python (e.g., using scipy.optimize.minimize)
        raise NotImplementedError("Mode calculation is not implemented")

    def trigonometric_moment(self, n):
        return self.trigonometric_moment_numerical(n)

    def integrate(self, left=None, right=None):
        if left is None:
            left = np.zeros(self.dim)
        if right is None:
            right = 2 * np.pi * np.ones(self.dim)

        assert left.shape == (self.dim,)
        assert right.shape == (self.dim,)

        return self.integrate_numerically(left, right)

    def mean_2dimD(self):
        m = self.trigonometric_moment_numerical(1)
        mu = np.vstack((m.real, m.imag))
        return mu

    def sample_metropolis_hastings(
        self, n, proposal=None, start_point=None, burn_in=10, skipping=5
    ):
        if proposal is None:

            def proposal(x):
                return np.mod(x + np.random.randn(self.dim), 2 * np.pi)

        if start_point is None:
            start_point = self.mean_direction()

        s = super().sample_metropolis_hastings(
            n, proposal, start_point, burn_in, skipping
        )
        return s

    @staticmethod
    def setup_axis_circular(axis_name="x", ax=plt.gca()):
        ticks = [0, np.pi, 2 * np.pi]
        tick_labels = ["0", r"$\pi$", r"$2\pi$"]
        if axis_name == "x":
            ax.set_xlim(left=0, right=2 * np.pi)
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)
        elif axis_name == "y":
            ax.set_ylim(left=0, right=2 * np.pi)
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels)
        elif axis_name == "z":
            ax.set_zlim(left=0, right=2 * np.pi)
            ax.set_zticks(ticks)
            ax.set_zticklabels(tick_labels)
        else:
            raise ValueError("invalid axis")
