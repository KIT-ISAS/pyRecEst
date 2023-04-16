import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import nquad, quad

from .abstract_periodic_distribution import AbstractPeriodicDistribution


class AbstractHypertoroidalDistribution(AbstractPeriodicDistribution):
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

    def integral(self, left=None, right=None):
        if left is None:
            left = np.zeros(self.dim)
        if right is None:
            right = 2 * np.pi * np.ones(self.dim)

        assert left.shape == (self.dim,)
        assert right.shape == (self.dim,)

        return self.integral_numerical(left, right)

    def integral_numerical(self, left, right):
        if self.dim == 1:
            result, _ = quad(self.pdf, left, right)
        elif self.dim == 2:
            result, _ = nquad(
                lambda x, y: self.pdf(np.array([x, y])),
                [[left[0], right[0]], [left[1], right[1]]],
            )
        elif self.dim == 3:
            result, _ = nquad(
                lambda x, y, z: self.pdf(np.array([x, y, z])),
                [[left[0], right[0]], [left[1], right[1]], [left[2], right[2]]],
            )
        else:
            raise ValueError(
                "Numerical moment calculation for this dimension is currently not supported"
            )
        return result

    def trigonometric_moment_numerical(self, n):
        if self.dim == 1:

            def fun_1d(x):
                return self.pdf(x) * np.exp(1j * n * x)

            m, _ = quad(fun_1d, 0, 2 * np.pi)

        elif self.dim == 2:
            m = np.zeros(2, dtype=complex)

            def fun1_2d(x, y):
                return self.pdf([x, y]) * np.exp(1j * n * x)

            def fun2_2d(x, y):
                return self.pdf([x, y]) * np.exp(1j * n * y)

            m[0], _ = nquad(fun1_2d, [(0, 2 * np.pi), (0, 2 * np.pi)])
            m[1], _ = nquad(fun2_2d, [(0, 2 * np.pi), (0, 2 * np.pi)])

        elif self.dim == 3:
            m = np.zeros(3, dtype=complex)

            def fun1_3d(x, y, z):
                return self.pdf([x, y, z]) * np.exp(1j * n * x)

            def fun2_3d(x, y, z):
                return self.pdf([x, y, z]) * np.exp(1j * n * y)

            def fun3_3d(x, y, z):
                return self.pdf([x, y, z]) * np.exp(1j * n * z)

            m[0], _ = nquad(fun1_3d, [(0, 2 * np.pi), (0, 2 * np.pi)])
            m[1], _ = nquad(fun2_3d, [(0, 2 * np.pi), (0, 2 * np.pi)])
            m[2], _ = nquad(fun3_3d, [(0, 2 * np.pi), (0, 2 * np.pi)])

        else:
            raise NotImplementedError(
                "Numerical moment calculation for this dimension is currently not supported"
            )

        return m

    def entropy(self):
        return self.entropy_numerical()

    def entropy_numerical(self):
        # Calculates the entropy numerically
        #
        # Returns:
        #   e (scalar)
        #       entropy of the distribution

        def integrand_1d(x):
            pdf_val = self.pdf(x)
            return pdf_val * np.log(pdf_val)

        def integrand_2d(x, y):
            xy = np.vstack([x, y])
            pdf_val = self.pdf(xy)
            return pdf_val * np.log(pdf_val)

        def integrand_3d(x, y, z):
            xyz = np.vstack([x, y, z])
            pdf_val = self.pdf(xyz)
            return pdf_val * np.log(pdf_val)

        if self.dim == 1:
            e, _ = nquad(integrand_1d, [[0, 2 * np.pi]])
        elif self.dim == 2:
            e, _ = nquad(integrand_2d, [[0, 2 * np.pi], [0, 2 * np.pi]])
        elif self.dim == 3:
            e, _ = nquad(integrand_3d, [[0, 2 * np.pi], [0, 2 * np.pi], [0, 2 * np.pi]])
        else:
            raise NotImplementedError(
                "Numerical moment calculation for this dimension is currently not supported."
            )

        return -e

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
